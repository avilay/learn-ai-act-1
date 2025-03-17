import os.path as path

import gym
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rl.tests.frozen_lake_policy import build_best_policy
from rl.valfuncs.hyperparams import Hyperparams
from rl.valfuncs.model_free import gen_dataset


class ValueFunc(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2)
        self.out = torch.nn.Linear(2, 1)
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        self.hparams = kwargs.get("hparams")
        self.trainds = kwargs.get("trainds")
        self.valds = kwargs.get("valds")

    @classmethod
    def for_training(cls, hparams):
        fl = gym.make("FrozenLake-v0")
        policy = build_best_policy(fl)
        trainds = gen_dataset(fl, policy, hparams.num_train_steps)
        valds = gen_dataset(fl, policy, hparams.num_val_steps)
        return cls(hparams=hparams, trainds=trainds, valds=valds)

    @classmethod
    def for_inference(cls):
        return cls()

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return self.out(x)

    def training_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        metrics = {"train_loss": loss.detach()}
        return {"loss": loss, "log": metrics, "progress_bar": metrics}

    def validation_step(self, batch, batch_num):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        metrics = {"val_loss": loss.detach()}
        return metrics

    def validation_end(self, outputs):
        avg_loss = torch.stack([output["val_loss"] for output in outputs]).mean()
        rmse = torch.sqrt(avg_loss)
        metrics = {"val_loss": avg_loss, "val_rmse": rmse}
        return {"val_loss": avg_loss, "log": metrics, "progress_bar": metrics}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.trainds, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.valds, batch_size=100)


def main():
    tblogs = path.expanduser("~/mldata/tblogs/frozen-lake")
    hparams = Hyperparams(
        batch_size=8, epochs=10, lr=0.01, num_train_steps=10_000, num_val_steps=1000
    )
    trainer = pl.Trainer(default_save_path=tblogs, max_nb_epochs=hparams.epochs)
    model = ValueFunc.for_training(hparams)
    trainer.fit(model)


main()
