import pickle
import warnings
from pathlib import Path

import click
import pytorch_lightning as pl
import torch as t
import torchmetrics as tm
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.utils.data import DataLoader

import common
from datagen import multiclass_classification
from omegaconf import OmegaConf


class MyDataModule(pl.LightningDataModule):
    def __init__(self, hp=None):
        super().__init__()
        self.hparams = hp
        self._dataroot = Path.cwd() / "data" / "multiclass"
        self._train_filename = self._dataroot / "trainset.pkl"
        self._val_filename = self._dataroot / "valset.pkl"
        self._test_filename = self._dataroot / "testset.pkl"
        self._trainset, self._valset, self._testset = None, None, None

    def prepare_data(self):
        self._dataroot.mkdir(parents=True, exist_ok=True)
        if not (
            self._train_filename.exists()
            and self._val_filename.exists()
            and self._test_filename.exists()
        ):
            print("Generating new datasets")
            trainset, valset, testset = multiclass_classification(n_samples=100_000)
            with open(self._train_filename, "wb") as f:
                pickle.dump(trainset, f)
            with open(self._val_filename, "wb") as f:
                pickle.dump(valset, f)
            with open(self._test_filename, "wb") as f:
                pickle.dump(testset, f)

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            with open(self._train_filename, "rb") as f:
                self._trainset = pickle.load(f)
            with open(self._val_filename, "rb") as f:
                self._valset = pickle.load(f)
        elif stage == "test":
            with open(self._test_filename, "rb") as f:
                self._testset = pickle.load(f)
        else:
            raise RuntimeError(f"Unknown stage {stage}!")

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


class MulticlassClassifier(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.fc1 = t.nn.Linear(20, 16)
        self.fc2 = t.nn.Linear(16, 8)
        self.fc3 = t.nn.Linear(8, 5)

        self.loss_fn = t.nn.CrossEntropyLoss()
        self.save_hyperparameters(hp)

    def accuracy(self, logits, y):
        y_hat = t.argmax(logits, dim=1)
        return tm.functional.accuracy(y_hat, y)

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        x = t.nn.functional.relu(self.fc1(x))
        x = t.nn.functional.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})


@click.group()
def main():
    pass


@main.command()
@click.argument("project")
@click.option("--run-name", default="", help="Name of this run.")
@click.option("--hparams", default="multiclass.yml", help="Hyperparams yml file.")
def train(project, run_name, hparams):
    hp = OmegaConf.load(hparams)
    model = MulticlassClassifier(hp)
    data = MyDataModule(hp)
    common.train(project=project, run_name=run_name, model=model, data=data, hparams=hp)


@main.command()
@click.argument("project")
@click.argument("run_name")
def test(project, run_name):
    common.test(project, run_name, MulticlassClassifier, MyDataModule)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=".*data_loading.*")

    main()
