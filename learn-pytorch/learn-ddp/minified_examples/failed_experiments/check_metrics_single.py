import csv
import glob
import pickle
import re
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_traceback
import pytorch_lightning as pl
import torch as t
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy

pretty_traceback.install()

DATAROOT = Path.home() / "mldata" / "binclass" / "10K"
RUNROOT = Path.home() / "mlruns" / "check_metrics_single"

# Hyperparmas
LR = 0.5
BATCH_SIZE = 32
N_EPOCHS = 3


class MyDataModule(pl.LightningDataModule):
    def __init__(self, dataroot, batch_size):
        super().__init__()
        self._dataroot = dataroot
        self._trainset, self._valset, self._testset = None, None, None

        self.save_hyperparameters(ignore=["dataroot"])

    def prepare_data(self):
        """Prepare datasets on rank:0 trainer

        Pickles train.csv, val.csv, and test.csv into tensor datasets that can be loaded
        by all other trainers later.
        """
        for name in ["train", "val", "test"]:
            filename = self._dataroot / f"{name}.csv"
            data = pd.read_csv(filename).values
            x = data[:, :-1].astype(np.float32)
            y = data[:, -1].astype(np.int8)

            ds = TensorDataset(t.from_numpy(x), t.from_numpy(y))
            pklfile = self._dataroot / f"{name}.pkl"
            with open(pklfile, "wb") as f:
                pickle.dump(ds, f)

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            pklfile = self._dataroot / "train.pkl"
            with open(pklfile, "rb") as f:
                self._trainset = pickle.load(f)

            pklfile = self._dataroot / "val.pkl"
            with open(pklfile, "rb") as f:
                self._valset = pickle.load(f)
        else:
            pklfile = self._dataroot / "test.pkl"
            with open(pklfile, "rb") as f:
                self._testset = pickle.load(f)

    def train_dataloader(self):
        return DataLoader(
            self._trainset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._valset, batch_size=self.hparams.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self._testset, batch_size=self.hparams.batch_size, shuffle=False
        )


class Net(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()

        self.fc = t.nn.Linear(5, 1)

        self.loss_fn = t.nn.BCELoss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim = t.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return optim

    def forward(self, inputs):
        probs = t.sigmoid(self.fc(inputs))
        return t.squeeze(probs, dim=1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.to(t.float32))
        return {"loss": loss, "outputs": outputs.detach()}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.to(t.float32))
        return {"loss": loss, "outputs": outputs.detach()}


class MetricsLoggerDebug(pl.Callback):
    """
    Here I am taking 2 sets of `tm.Accuracy`, one is
    used by the logger where the logger is calling `compute` and `reset` internally,
    the second is where I am doing this manually and printing out the results. Finally, there is
    a third set of metrics that I calculate completely by hand.
    All three sets of numbers should match.
    """

    def __init__(self, runroot):
        super().__init__()

        self.runroot = runroot

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.train_acc_dbg = Accuracy()
        self.val_acc_dbg = Accuracy()

        self.train_y, self.train_y_hat = [], []
        self.val_y, self.val_y_hat = [], []

        self.train_epoch = -1
        self.val_epoch = -1

    def on_train_epoch_start(self, trainer, model):
        self.train_epoch += 1

        self.train_y, self.train_y_hat = [], []
        self.train_acc_dbg.reset()

    def on_train_batch_end(self, trainer, model, retval, batch, batch_idx):
        loss = retval["loss"]
        outputs = retval["outputs"]
        targets = batch[1]

        self.train_acc(outputs, targets)
        model.log("train_step_loss", loss)
        model.log("loss", {"train": loss}, on_step=False, on_epoch=True)
        model.log("acc", {"train": self.train_acc}, on_step=False, on_epoch=True)

        self.train_acc_dbg(outputs, targets)

        y = targets.numpy()
        y_hat = (outputs.detach().numpy() > 0.5).astype(int)
        self.train_y.append(list(y))
        self.train_y_hat.append(list(y_hat))

    def on_validation_epoch_start(self, trainer, model):
        self.val_epoch += 1

        self.val_y, self.val_y_hat = [], []
        self.val_acc_dbg.reset()

    def on_validation_batch_end(
        self, trainer, model, retval, batch, batch_idx, dataloader_idx
    ):
        loss = retval["loss"]
        outputs = retval["outputs"]
        targets = batch[1]

        self.val_acc(outputs, targets)
        model.log("loss", {"val": loss})
        model.log("acc", {"val": self.val_acc})

        self.val_acc_dbg(outputs, targets)

        y = targets.numpy()
        y_hat = (outputs.detach().numpy() > 0.5).astype(int)
        self.val_y.append(list(y))
        self.val_y_hat.append(list(y_hat))

    def on_validation_epoch_end(self, trainer, model):
        y = np.array(reduce(lambda acc, ary: acc + ary, self.val_y, []))
        y_hat = np.array(reduce(lambda acc, ary: acc + ary, self.val_y_hat, []))
        acc = np.mean((y_hat == y).astype(int))

        acc_dbg = self.val_acc_dbg.compute()

        with open(self.runroot / "manual.txt", "at") as f:
            print(
                f"Epoch {self.val_epoch}: Val accuracy = {acc}, {acc_dbg.item()}",
                file=f,
            )

    def on_train_epoch_end(self, trainer, model):
        y = np.array(reduce(lambda acc, ary: acc + ary, self.train_y, []))
        y_hat = np.array(reduce(lambda acc, ary: acc + ary, self.train_y_hat, []))
        acc = np.mean((y_hat == y).astype(int))

        acc_dbg = self.train_acc_dbg.compute()

        with open(self.runroot / "manual.txt", "at") as f:
            print(
                f"Epoch {self.train_epoch}: Train accuracy = {acc}, {acc_dbg.item()}",
                file=f,
            )


@dataclass
class Metric:
    epoch: int = -1
    train: float = float("inf")
    train_manual: float = float("inf")
    train_dbg: float = float("inf")
    val: float = float("inf")
    val_manual: float = float("inf")
    val_dbg: float = float("inf")

    def __str__(self):
        ret = f"Epoch {self.epoch}\n"
        ret += f"\tTrain: manual = {self.train_manual}, lit = {self.train}, dbg = {self.train_dbg}\n"
        ret += (
            f"\tVal: manual = {self.val_manual}, lit = {self.val}, dbg = {self.val_dbg}"
        )
        return ret


def check_results():
    epochs = {}

    with open(RUNROOT / "manual.txt") as f:
        for line in f:
            m = re.search(r"Epoch (\d+): (\w+) accuracy = (\d+\.\d+), (\d+\.\d+)", line)
            epoch = int(m.group(1))
            stage = m.group(2).lower().strip()
            manual = float(m.group(3))
            dbg = float(m.group(4))
            if epoch not in epochs:
                epochs[epoch] = Metric(epoch=epoch)
            metric = epochs[epoch]
            if stage == "val":
                metric.val_manual = manual
                metric.val_dbg = dbg
            elif stage == "train":
                metric.train_manual = manual
                metric.train_dbg = dbg
            else:
                RuntimeError("KA-BOOM!!")

    # Get the latest version of run
    pattern = str(RUNROOT / "version*")
    paths = sorted([(Path(p).stat().st_atime, Path(p)) for p in glob.glob(pattern)])
    latest_path = paths[-1][1] / "metrics.csv"
    with open(latest_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["acc"]:
                epoch = int(row["epoch"])
                m = re.search(r"{'(\w+)': (\d+\.\d+)", row["acc"])
                stage = m.group(1).lower().strip()
                acc = float(m.group(2))
                metric = epochs[epoch]
                if stage == "val":
                    metric.val = acc
                elif stage == "train":
                    metric.train = acc
                else:
                    raise RuntimeError("KA-BOOM!")

    print("\n\nRESULTS")
    for epoch in epochs.values():
        fail = False
        print(epoch)
        if not np.isclose(epoch.train_manual, epoch.train):
            print("FAIL: train manual is not close to lit!")
            fail = True
        if not np.isclose(epoch.train_manual, epoch.train_dbg):
            print("FAIL: train manual is not close to dbg!")
            fail = True
        if not np.isclose(epoch.val_manual, epoch.val):
            print("FAIL: val manual is not close to lit!")
            fail = True
        if not np.isclose(epoch.val_manual, epoch.val_dbg):
            print("FAIL: val manual is not close to dbg!")
            fail = True
        if not fail:
            print("PASS\n")


def train():
    data = MyDataModule(DATAROOT, BATCH_SIZE)
    logger = CSVLogger(save_dir=RUNROOT.parent, name=RUNROOT.name)
    trainer = pl.Trainer(
        default_root_dir=RUNROOT,
        max_epochs=N_EPOCHS,
        callbacks=[
            MetricsLoggerDebug(RUNROOT),
        ],
        num_sanity_val_steps=0,
        logger=logger,
    )
    net = Net(LR)
    trainer.fit(net, data)
    check_results()


if __name__ == "__main__":
    train()
