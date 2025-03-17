import datagen
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import click
import torch as t
import common
import warnings
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
import torchmetrics as tm


class MyDataModule(pl.LightningDataModule):
    def __init__(self, hp=None):
        super().__init__()
        self._dataroot = Path.cwd() / "data" / "regression"
        self._train_filename = self._dataroot / "trainset.pkl"
        self._val_filename = self._dataroot / "valset.pkl"
        self._test_filename = self._dataroot / "testset.pkl"

        self._trainset, self._valset, self._testset = None, None, None

        self.hparams = hp

    def prepare_data(self):
        self._dataroot.mkdir(parents=True, exist_ok=True)
        if not (
            self._train_filename.exists()
            and self._val_filename.exists()
            and self._test_filename.exists()
        ):
            trainset, valset, testset = datagen.regression(n_samples=100_000)
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

    def train_dataloader(self):
        return DataLoader(
            self._trainset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self._valset, batch_size=5000)

    def test_dataloader(self):
        return DataLoader(self._testset, batch_size=5000)


class Regressor(pl.LightningModule):
    def __init__(self, hp):
        super().__init__()
        self.fc1 = t.nn.Linear(5, 3)
        self.fc2 = t.nn.Linear(3, 1)

        self.save_hyperparameters(hp)
        self.loss_fn = t.nn.MSELoss()

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def mean_abs_err(self, y_hat, y):
        return tm.functional.mean_absolute_error(y_hat, y)

    def forward(self, x):
        x = t.nn.functional.relu(self.fc1(x))
        y_hat = self.fc2(x)
        return t.squeeze(y_hat, dim=1)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        mae = self.mean_abs_err(y_hat, y)
        return loss, mae

    def training_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch)
        self.log("train_loss_step", loss)
        self.log_dict(
            {"train_loss": loss, "train_mae": mae}, on_step=False, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch)
        self.log_dict({"val_loss": loss, "val_mae": mae})

    def test_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch)
        self.log_dict({"test_loss": loss, "test_mae": mae})


@click.group()
def main():
    pass


@main.command()
@click.argument("project")
@click.option("--run-name", default="", help="Name of this run.")
@click.option("--hparams", default="regressor.yml", help="Hyperparams yml file.")
def train(project, run_name, hparams):
    common.train(
        project=project,
        run_name=run_name,
        model_cls=Regressor,
        data_cls=MyDataModule,
        hparams_yml=hparams,
    )


@main.command()
@click.argument("project")
@click.argument("run_name")
def test(project, run_name):
    common.test(project, run_name, Regressor, MyDataModule)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=".*data_loading.*")

    main()
