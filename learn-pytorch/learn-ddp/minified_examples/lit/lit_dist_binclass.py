"""
Use 2 GPUs to train, but a single GPU:0 for test.

python lit_dist_binclass.py --train
python lit_dist_binclass.py --test --ckpt=$CKPT
"""
from pathlib import Path

import click
import pretty_traceback
import torch as t
import torch.distributed as dist
from bindata import BinDataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.plugins import DDPPlugin
from torch.nn import BCELoss, Linear
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import Accuracy

pretty_traceback.install()

# Hyperparams
LR = 0.003
N_EPOCHS = 10
BATCH_SIZE = (
    16  # Halving the batch size because each GPU will work with batches of this size
)
# BATCH_SIZE = 32

RUNROOT = Path.home() / "mlruns" / "lit_dist_binclass"
DATAROOT = "file://" + str(Path.home() / "mldata" / "binclass" / "1M")
N_PARTS = 10


class MyDataModule(LightningDataModule):
    def __init__(self, dataroot, n_parts, batch_size):
        super().__init__()
        self._trainsets = []
        self._valsets = []
        self._testdl = None
        self._dataroot = dataroot
        self._n_parts = n_parts
        self.save_hyperparameters(ignore=["dataroot", "n_parts"])

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            trainsets_0, valsets_0 = BinDataset.load_train_val_partitioned(
                self._dataroot, range(self._n_parts // 2)
            )
            self._trainsets.append(ConcatDataset(trainsets_0))
            self._valsets.append(ConcatDataset(valsets_0))

            trainsets_1, valsets_1 = BinDataset.load_train_val_partitioned(
                self._dataroot, range(self._n_parts // 2, self._n_parts)
            )
            self._trainsets.append(ConcatDataset(trainsets_1))
            self._valsets.append(ConcatDataset(valsets_1))
        else:
            testset = BinDataset.load_test_single(self._dataroot)
            self._testdl = DataLoader(testset, batch_size=1000, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            self._trainsets[dist.get_rank()],
            batch_size=self.hparams.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valsets[dist.get_rank()], batch_size=1000, shuffle=False
        )

    def test_dataloader(self):
        return self._testdl


class Net(LightningModule):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.fc1 = Linear(20, 32)
        self.fc2 = Linear(32, 1)
        self.loss_fn = BCELoss()

        self.train_acc = Accuracy(compute_on_step=False)
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        y_hat = t.sigmoid(x)
        return y_hat.squeeze(dim=1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.log("train_step_loss", loss)
        self.log("loss", {"train_loss": loss}, on_step=False, on_epoch=True)
        return {"loss": loss, "outputs": outputs.detach(), "targets": targets}

    def training_step_end(self, out):
        self.train_acc(out["outputs"], out["targets"])
        self.log("acc", {"train_acc": self.train_acc}, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.log("loss", {"val_loss": loss})
        return {"outputs": outputs, "targets": targets}

    def validation_step_end(self, out):
        self.val_acc(out["outputs"], out["targets"])
        self.log("acc", {"val_acc": self.val_acc})

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        return {"outputs": outputs, "targets": targets}

    def test_step_end(self, out):
        self.test_acc(out["outputs"], out["targets"])
        self.log("test_acc", self.test_acc)


@click.command()
@click.option("--train/--test", help="Whether to train or test.", default=True)
@click.option("--ckpt", help="Checkpoint path", default="")
def main(train, ckpt):
    dm = MyDataModule(dataroot=DATAROOT, batch_size=BATCH_SIZE, n_parts=N_PARTS)
    logger = CSVLogger(save_dir=RUNROOT.parent, name=RUNROOT.name)
    if train:
        trainer = Trainer(
            default_root_dir=RUNROOT,
            max_epochs=N_EPOCHS,
            progress_bar_refresh_rate=10,
            gpus=-1,
            replace_sampler_ddp=False,
            strategy=DDPPlugin(find_unused_parameters=False),
            logger=logger,
        )
        model = Net(learning_rate=LR)
        trainer.fit(model, dm)

        # model.current_epoch and model.global_step don't seem to work
    else:
        trainer = Trainer(
            default_root_dir=RUNROOT,
            progress_bar_refresh_rate=0,
            gpus=[0],
            replace_sampler_ddp=False,
            logger=logger,
        )
        model = Net.load_from_checkpoint(ckpt)
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
