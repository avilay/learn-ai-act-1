"""
3 seconds for standard imports
13 seconds for pytorch imports
9 seconds for lightning imports

As can be seen from the outputs, Lighthing will automatically shuffle and add the
distributed sampler to the data loader even though I didn't explicitly didn't do that.

$ python toylit.py --dload
Batch 0
tensor([[ 0.,  1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  8.,  9.],
        [10., 11., 12., 13., 14.]])

Batch 1
tensor([[15., 16., 17., 18., 19.],
        [20., 21., 22., 23., 24.],
        [25., 26., 27., 28., 29.]])

Batch 2
tensor([[30., 31., 32., 33., 34.],
        [35., 36., 37., 38., 39.],
        [40., 41., 42., 43., 44.]])

Batch 3
tensor([[45., 46., 47., 48., 49.],
        [50., 51., 52., 53., 54.],
        [55., 56., 57., 58., 59.]])

Batch 4
tensor([[60., 61., 62., 63., 64.],
        [65., 66., 67., 68., 69.],
        [70., 71., 72., 73., 74.]])

Batch 5
tensor([[75., 76., 77., 78., 79.],
        [80., 81., 82., 83., 84.],
        [85., 86., 87., 88., 89.]])

Batch 6
tensor([[90., 91., 92., 93., 94.],
        [95., 96., 97., 98., 99.]])

$ python toylit.py --train
[1](3312): TRAIN: 0:
tensor([[25., 26., 27., 28., 29.],
        [95., 96., 97., 98., 99.],                                                                                                    [0/558]
        [70., 71., 72., 73., 74.]], device='cuda:1')

[0](3311): TRAIN: 0:
tensor([[20., 21., 22., 23., 24.],
        [65., 66., 67., 68., 69.],
        [35., 36., 37., 38., 39.]], device='cuda:0')

[1](3312): TRAIN: 1:
tensor([[30., 31., 32., 33., 34.],
        [85., 86., 87., 88., 89.],
        [10., 11., 12., 13., 14.]], device='cuda:1')

[0](3311): TRAIN: 1:
tensor([[15., 16., 17., 18., 19.],
        [45., 46., 47., 48., 49.],
        [55., 56., 57., 58., 59.]], device='cuda:0')

[1](3312): TRAIN: 2:
tensor([[90., 91., 92., 93., 94.],
        [60., 61., 62., 63., 64.],
        [40., 41., 42., 43., 44.]], device='cuda:1')

[0](3311): TRAIN: 2:
tensor([[80., 81., 82., 83., 84.],
        [50., 51., 52., 53., 54.],
        [75., 76., 77., 78., 79.]], device='cuda:0')

[1](3312): TRAIN: 3:
tensor([[0., 1., 2., 3., 4.]], device='cuda:1')

[0](3311): TRAIN: 3:
tensor([[5., 6., 7., 8., 9.]], device='cuda:0')

"""

import os

import click
import numpy as np
import pretty_traceback
import pytorch_lightning as pl
import torch as t
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from yachalk import chalk
from pytorch_lightning.plugins import DDPPlugin

pretty_traceback.install()

color_chalk = [chalk.yellow_bright, chalk.blue_bright]


def dist_print(text):
    rank = dist.get_rank() if dist.is_initialized() else -1
    pid = os.getpid()
    print(color_chalk[rank](f"[{rank}]({pid}): {text}"))


class MyMappedDataset(Dataset):
    def __init__(self, n=5, m=10):
        self._x = np.arange(n * m).reshape(m, n).astype(np.float32)
        self._y = np.random.choice([0, 1], size=m, p=[0.7, 0.3]).astype(np.int8)

    def __getitem__(self, idx):
        return self._x[idx], self._y[idx]

    def __len__(self):
        return self._x.shape[0]


class MyData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self._trainset = MyMappedDataset(n=5, m=20)
        self._valset = MyMappedDataset(n=5, m=10)

    def train_dataloader(self):
        # dist_print("You can see my rank.")
        return DataLoader(self._trainset, batch_size=3, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self._valset, batch_size=3, shuffle=False)


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        y_hat = t.sigmoid(self.fc(x))
        return t.squeeze(y_hat, dim=1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # dist_print(f"TRAIN: {batch_idx}:\n{inputs}")
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        # dist_print(f"VAL: {batch_idx}:\n{inputs}")
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.type_as(outputs))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optim = t.optim.Adam(self.parameters(), lr=0.01)
        return optim


@click.command()
@click.option(
    "--train/--dload",
    help="Run the train loop or just iterate through the data loader.",
    required=True,
)
def main(train):
    dist_print("MAIN")
    if train:
        known_keys = set()
        with open("/home/avilay/oskeys.txt", "rt") as f:
            for line in f:
                line = line.strip()
                known_keys.add(line)
        dist_print("Printing new keys-")
        for key, val in os.environ.items():
            if key not in known_keys:
                dist_print(f"ENV: {key}={val}")
        model = Net()
        data = MyData()
        trainer = pl.Trainer(
            max_epochs=1,
            progress_bar_refresh_rate=0,
            num_sanity_val_steps=0,
            gpus=-1,
            strategy=DDPPlugin(find_unused_parameters=False),
        )

        if dist.is_initialized():
            dist_print("I am initialized!")
        else:
            dist_print("Still not initalized.")

        trainer.fit(model, data)

    else:
        data = MyData()
        dl = data.train_dataloader()
        for i, batch in enumerate(dl):
            x, y = batch
            print(f"\nBatch {i}")
            print(x)


if __name__ == "__main__":
    main()
