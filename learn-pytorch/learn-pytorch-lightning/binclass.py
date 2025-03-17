import logging
import pickle
import warnings
from pathlib import Path

import click
from omegaconf.omegaconf import OmegaConf
import pytorch_lightning as pl
import torch as t
import torchmetrics as tm
from torch.utils.data import DataLoader

import common
from datagen import binary_classification


class MyDataModule(pl.LightningDataModule):
    def __init__(self, hp=None):
        super().__init__()
        self._trainset, self._valset, self._testset = None, None, None
        self._datadir = Path.cwd() / "data" / "binclass"
        self._train_filename = self._datadir / "trainset.pkl"
        self._val_filename = self._datadir / "valset.pkl"
        self._test_filename = self._datadir / "testset.pkl"
        self.hparams = hp

    def prepare_data(self):
        self._datadir.mkdir(parents=True, exist_ok=True)

        if not (
            self._train_filename.exists()
            and self._val_filename.exists()
            and self._test_filename.exists()
        ):
            print("Generating new training data")
            trainset, valset, testset = binary_classification(n_samples=100_000)
            with open(self._train_filename, "wb") as trainfile:
                pickle.dump(self._trainset, trainfile)
            with open(self._val_filename, "wb") as valfile:
                pickle.dump(self._valset, valfile)
            with open(self.test_filename, "wb") as testfile:
                pickle.dump(self._testset, testfile)

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            with open(self._train_filename, "rb") as trainfile:
                self._trainset = pickle.load(trainfile)
            with open(self._val_filename, "rb") as valfile:
                self._valset = pickle.load(valfile)
        elif stage == "test":
            with open(self._test_filename, "rb") as testfile:
                self._testset = pickle.load(testfile)

    def train_dataloader(self):
        train_dl = DataLoader(
            self._trainset, batch_size=self.hparams.batch_size, shuffle=True
        )
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self._valset, batch_size=5000)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self._testset, batch_size=5000)
        return test_dl

    def teardown(self, stage):
        pass


class BinaryClassifier(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.fc1 = t.nn.Linear(20, 8)
        self.fc2 = t.nn.Linear(8, 1)
        self.loss_fn = t.nn.BCELoss()

        # The following line with do two things -
        #   1. Create self.hparams.lr and self.hparams.true_cutoff
        #   2. Log the hyper params
        self.save_hyperparameters(hp)

    def accuracy(self, p, y):
        y_hat = (p > self.hparams.true_cutoff).to(t.int)
        return tm.functional.accuracy(y_hat, y)

    def forward(self, x):
        x = t.nn.functional.relu(self.fc1(x))
        y_hat = t.sigmoid(self.fc2(x))
        return t.squeeze(y_hat, dim=1)

    def _shared_step(self, batch):
        x, y = batch
        p = self(x)
        loss = self.loss_fn(p, y.to(t.float32))
        acc = self.accuracy(p, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_los_step", loss)
        self.log_dict(
            {"train_acc": acc, "train_loss": loss}, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})

    def configure_optimizers(self):
        optim = t.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optim


@click.group()
def main():
    pass


@main.command()
@click.argument("project")
@click.option("--run-name", default="", help="Name of this run.")
@click.option("--hparams", default="binclass.yml", help="Hyperparams yml file.")
def train(project, run_name, hparams):
    hp = OmegaConf.load(hparams)
    model = BinaryClassifier(hp)
    data = MyDataModule(hp)
    common.train(project=project, run_name=run_name, model=model, data=data, hparams=hp)


@main.command()
@click.argument("project")
@click.argument("run_name")
def test(project, run_name):
    common.test(project, run_name, BinaryClassifier, MyDataModule)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    logging.getLogger("wandb").setLevel(logging.ERROR)
    main()
