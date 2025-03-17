from pathlib import Path

import click
import pretty_traceback
import torch as t
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.nn import BCELoss, Linear
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from pytorch_lightning.loggers import CSVLogger

from bindata import BinDataset

pretty_traceback.install()

# Hyperparams
LR = 0.003
N_EPOCHS = 10
BATCH_SIZE = 16

RUNROOT = Path.home() / "mlruns" / "lit_single_binclass"
# DATAROOT = "file:///Users/avilay/mldata/binclass"
DATAROOT = "file:///home/avilay/mldata/binclass"


class MyDataModule(LightningDataModule):
    def __init__(self, dataroot, batch_size):
        super().__init__()
        self._traindl = None
        self._valdl = None
        self._testdl = None
        self._dataroot = dataroot
        self.save_hyperparameters(ignore="dataroot")

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            trainset, valset = BinDataset.load_train_val_single(self._dataroot)
            self._traindl = DataLoader(
                trainset, batch_size=self.hparams.batch_size, shuffle=True,
            )
            self._valdl = DataLoader(valset, batch_size=1000, shuffle=False)
        else:
            testset = BinDataset.load_test_single(self._dataroot)
            self._testdl = DataLoader(testset, batch_size=1000, shuffle=False)

    def train_dataloader(self):
        return self._traindl

    def val_dataloader(self):
        return self._valdl

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

    def _shared_step(self, batch, acc_metric):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        acc_metric(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.train_acc)
        self.log("train_step_loss", loss)
        self.log("loss", {"train_loss": loss}, on_step=False, on_epoch=True)
        self.log("acc", {"train_acc": self.train_acc}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.val_acc)
        self.log("loss", {"val_loss": loss})
        self.log("acc", {"val_acc": self.val_acc})

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, self.test_acc)
        self.log("test_acc", self.test_acc)


@click.command()
@click.option("--train/--test", help="Whether to train or test.", default=True)
@click.option("--ckpt", help="Checkpoint path", default="")
def main(train, ckpt):
    dm = MyDataModule(dataroot=DATAROOT, batch_size=BATCH_SIZE)
    logger = CSVLogger(save_dir=RUNROOT.parent, name=RUNROOT.name)
    if train:
        trainer = Trainer(
            default_root_dir=RUNROOT,
            max_epochs=N_EPOCHS,
            progress_bar_refresh_rate=10,
            logger=logger,
        )
        model = Net(learning_rate=LR)
        trainer.fit(model, dm)
        # ckpt_path = f"{trainer.log_dir}/checkpoints/epoch={trainer.current_epoch}-step={trainer.global_step-1}.ckpt"
        # print(f"\n\nCheckpoint saved at {ckpt_path}")
    else:
        trainer = Trainer(default_root_dir=RUNROOT, progress_bar_refresh_rate=0)
        model = Net.load_from_checkpoint(ckpt)
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
