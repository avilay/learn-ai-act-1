from omegaconf.omegaconf import OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from pathlib import Path
import torch as t
import torchmetrics as tm
import wandb
import click
import common


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, hp=None):
        super().__init__()
        self.hparams = hp
        self._dataroot = Path.home() / "mldata" / "mnist"
        self._xform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self._trainset, self._valset, self._testset = None, None, None

    def prepare_data(self):
        MNIST(self._dataroot, train=True, download=True)
        MNIST(self._dataroot, train=False, download=True)

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            trainvalset = MNIST(self._dataroot, train=True, transform=self._xform)
            self._trainset, self._valset = random_split(trainvalset, [55_000, 5_000])
        elif stage == "test":
            self._testset = MNIST(self._dataroot, train=False, transform=self._xform)
        else:
            raise ValueError(f"Unknown stage {stage}!")

    def train_dataloader(self):
        return DataLoader(
            self._trainset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self._valset, batch_size=1000)

    def test_dataloader(self):
        return DataLoader(self._testset, batch_size=1000)


class FeedForwardNet(pl.LightningModule):
    def __init__(self, hp, n_classes):
        super().__init__()

        self.fc1 = t.nn.Linear(28 * 28, hp.n_layer_1)
        self.fc2 = t.nn.Linear(hp.n_layer_1, hp.n_layer_2)
        self.fc3 = t.nn.Linear(hp.n_layer_2, n_classes)

        self.loss_fn = t.nn.CrossEntropyLoss()

        self.save_hyperparameters(hp)

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) --> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # feed forward
        x = t.nn.functional.relu(self.fc1(x))
        x = t.nn.functional.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        y_hat = t.argmax(logits, dim=1)
        acc = tm.functional.accuracy(y_hat, y)
        return loss, acc, y_hat

    def training_step(self, batch, batch_idx):
        loss, acc, _ = self._shared_step(batch)
        self.log("train_loss_step", loss)

        epoch_metrics = {"train_loss": loss, "train_acc": acc}
        self.log_dict(epoch_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, y_hat = self._shared_step(batch)
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics)
        return y_hat

    def test_step(self, batch, batch_idx):
        loss, acc, y_hat = self._shared_step(batch)
        metrics = {"test_loss": loss, "test_acc": acc}
        self.log_dict(metrics)
        return y_hat


class LogPredictionsCallback(pl.Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 0:
            print("APTG DEBUG: Inside callback")
            n = 20
            x, y = batch
            examples = []
            for x_i, y_i, y_pred in zip(x[:n], y[:n], outputs[:n]):
                caption = f"Ground Truth: {y_i}\nPrediction: {y_pred}"
                example = wandb.Image(x_i, caption=caption)
                examples.append(example)
            pl_module.logger.experiment.log({"examples": examples})


@click.group()
def main():
    pass


@main.command()
@click.argument("project")
@click.option("--run-name", default="", help="Name of this run.")
@click.option("--hparams", default="mnist.yml", help="Hyperparams yml file.")
def train(project, run_name, hparams):
    hp = OmegaConf.load(hparams)
    model = FeedForwardNet(hp, n_classes=10)
    mnist = MNISTDataModule(hp)

    common.train_new(
        project=project,
        run_name=run_name,
        model=model,
        data=mnist,
        hparams=hp,
        callbacks=[LogPredictionsCallback()],
        progress_bar_refresh_rate=0,
    )


@main.command()
@click.argument("project")
@click.argument("run_name")
def test(project, run_name):
    common.test(project, run_name, FeedForwardNet, MNISTDataModule)


@main.command()
def check():
    # Check that test accuracy is around 10% without any training
    hparams = OmegaConf.create(
        {
            "n_layer_1": 128,
            "n_layer_2": 256,
            "batch_size": 256,
            "lr": 1e-3,
            "n_epochs": 3,
        }
    )
    pl.Trainer().test(FeedForwardNet(hparams, n_classes=10), MNISTDataModule())


if __name__ == "__main__":
    main()
