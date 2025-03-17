from pathlib import Path

import click
import pretty_traceback
import torch as t
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Callback, Trainer
from torchmetrics import Accuracy

from yachalk import chalk
import torch.distributed as dist
import os
import torchvision as tv
from pytorch_lightning.callbacks import TQDMProgressBar

pretty_traceback.install()

# Hyperparams
N_EPOCHS = 7
BATCH_SIZE = 64
DROPOUTS = [0.25, 0.5]
MOMENTUM = 0.9
LR = 0.001

RUNROOT = Path.home() / "mlruns" / "lit_mnist_improved"
DATAROOT = Path.home() / "mldata" / "mnist"

color_chalk = [chalk.yellow_bright, chalk.blue_bright]


def dist_print(text):
    rank = dist.get_rank() if dist.is_initialized() else -1
    pid = os.getpid()
    print(color_chalk[rank](f"[{rank}]({pid}): {text}"))
    # print(f"[{rank}]({pid}): {text}")


class Net(LightningModule):
    def __init__(self, learning_rate, momentum, dropouts):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout2d(dropouts[0])
        self.dropout2 = t.nn.Dropout2d(dropouts[1])
        self.fc1 = t.nn.Linear(9216, 128)
        self.fc2 = t.nn.Linear(128, 10)

        self.loss_fn = t.nn.NLLLoss()

        self._test_acc = Accuracy(compute_on_step=False)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def configure_optimizers(self):
        optim = t.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
        )
        return optim

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss, "outputs": outputs.detach()}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss, "outputs": outputs.detach()}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        return {"outputs": outputs.detach()}


class MetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self._train_acc = Accuracy(compute_on_step=False)
        self._val_acc = Accuracy(compute_on_step=False)
        self._test_acc = Accuracy(compute_on_step=False)

    def on_train_batch_end(self, trainer, pl_module, retval, batch, batch_idx):
        loss = retval["loss"]
        targets = batch[1]
        outputs = retval["outputs"]

        pl_module.log("train_step_loss", loss)
        pl_module.log("loss", {"train_loss": loss}, on_step=False, on_epoch=True)
        self._train_acc(outputs, targets)
        pl_module.log(
            "acc", {"train_acc": self._train_acc}, on_step=False, on_epoch=True
        )

    def on_validation_batch_end(
        self, trainer, pl_module, retval, batch, batch_idx, dataloader_idx
    ):
        loss = retval["loss"]
        targets = batch[1]
        outputs = retval["outputs"]

        pl_module.log("loss", {"val_loss": loss})
        self._val_acc(outputs, targets)
        pl_module.log("acc", {"val_acc": self._val_acc})

    def on_test_batch_end(
        self, trainer, pl_module, retval, batch, batch_idx, dataloader_idx
    ):
        outputs = retval["outputs"]
        targets = batch[1]
        self._test_acc(outputs, targets)
        pl_module.log("acc", {"test": self._test_acc})


class MnistDataModule(LightningDataModule):
    def __init__(self, dataroot, batch_size):
        super().__init__()
        self._trainset = None
        self._valset = None
        self._testset = None
        self._dataroot = dataroot
        self.save_hyperparameters(ignore=["dataroot"])

    def prepare_data(self):
        tv.datasets.MNIST(self._dataroot, train=True, download=True)
        tv.datasets.MNIST(self._dataroot, train=False, download=True)

    def setup(self, stage):
        xform = tv.transforms.Compose(
            [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
        )
        if stage == "fit" or stage == "validate":
            mnist = tv.datasets.MNIST(
                self._dataroot, download=False, train=True, transform=xform
            )
            train_size = int(len(mnist) * 0.9)
            self._trainset = t.utils.data.Subset(mnist, range(train_size))
            self._valset = t.utils.data.Subset(mnist, range(train_size, len(mnist)))
        elif stage == "test":
            self._testset = tv.datasets.MNIST(
                self._dataroot, train=False, transform=xform
            )

    def train_dataloader(self):
        return t.utils.data.DataLoader(
            self._trainset, batch_size=self.hparams.batch_size, num_workers=2
        )

    def val_dataloader(self):
        return t.utils.data.DataLoader(self._valset, batch_size=1000, num_workers=2)

    def test_dataloader(self):
        return t.utils.data.DataLoader(self._testset, batch_size=500)


@click.command()
@click.option("--train/--test", help="Whether to train or test", default=True)
@click.option("--ckpt", help="Checkpoint path", default="")
def main(train, ckpt):
    dm = MnistDataModule(dataroot=DATAROOT, batch_size=BATCH_SIZE)

    trainer = Trainer(
        default_root_dir=RUNROOT,
        max_epochs=N_EPOCHS,
        gpus=-1 if t.cuda.is_available() else None,
        strategy="ddp_find_unused_parameters_false" if t.cuda.is_available() else None,
        callbacks=[TQDMProgressBar(refresh_rate=10), MetricsLogger()],
    )
    if train:
        model = Net(learning_rate=LR, momentum=MOMENTUM, dropouts=DROPOUTS)
        trainer.fit(model, dm)
    else:
        model = Net.load_from_checkpoint(ckpt)
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
