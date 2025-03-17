"""
To train a dataset -
```
python single_binclass --train
```

To evaluate -
```
python single_binclass --test
```

Trainset has 7,000,000 instances (21,875 batches each with 32 instances)
Testset has 1,000,000 instances (100 batches each with 1000 instances)

On GCP
  * With CPU:
    - 1 minute per epoch
    - At the end of 2 epochs
      - Train: loss=0.216 acc=0.942                                                                               │
      - Val: loss=0.216 acc=0.943
      - Test accuracy: 0.943
  * With GPU:
    - 2 minutes per epoch
    - At the end of 2 epochs
      - Train: loss=0.218 acc=0.940
      - Val: loss=0.213 acc=0.943
      - Test accuracy: 0.942
"""
from pathlib import Path

import click
import pretty_traceback
import torch as t
import torchmetrics as tm
from tqdm import tqdm

from bindata import BinDataset

pretty_traceback.install()

# Hyperparams
LR = 0.003
N_EPOCHS = 10
BATCH_SIZE = 32

# DEVICE = t.device("cpu")
DEVICE = t.device("cuda:0")
DATAROOT = "file:///home/avilay/mldata/binclass"
# DATAROOT = "gs://avilabs-mldata/binclass"
# DATAROOT = "file:///Users/avilay/mldata/binclass"
CHECKPOINT = Path.home() / "mlruns" / "single_binclass_net.ckpt"


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(20, 32)
        self.fc2 = t.nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = t.nn.functional.relu(x)
        x = self.fc2(x)
        y_hat = t.sigmoid(x)
        return y_hat.squeeze(dim=1)


def training_loop(model, optim, loss_fn, traindl, valdl):
    for epoch in range(1, N_EPOCHS + 1):
        train_acc = tm.Accuracy().to(DEVICE)
        train_loss = tm.MeanMetric().to(DEVICE)

        model.train()
        with t.enable_grad():
            for inputs, targets in tqdm(traindl):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                # The standard 5-step training process
                optim.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets.to(t.float32))
                loss.backward()
                optim.step()

                train_acc(outputs, targets)
                train_loss(loss.detach().item())
        train_acc_value = train_acc.compute().item()
        train_loss_value = train_loss.compute().item()

        val_acc = tm.Accuracy(compute_on_step=False).to(DEVICE)
        val_loss = tm.MeanMetric(compute_on_step=False).to(DEVICE)

        model.eval()
        with t.no_grad():
            for inputs, targets in valdl:
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets.to(t.float32))

                val_acc(outputs, targets)
                val_loss(loss.detach().item())
        val_acc_value = val_acc.compute().item()
        val_loss_value = val_loss.compute().item()

        print(f"Epoch: {epoch}")
        print(f"Train: loss={train_loss_value:.3f} acc={train_acc_value:.3f}")
        print(f"Val: loss={val_loss_value:.3f} acc={val_acc_value:.3f}")
        print("\n")


def evaluate(model, testdl):
    acc = tm.Accuracy(compute_on_step=False).to(DEVICE)
    model.eval()
    with t.no_grad():
        for inputs, targets in tqdm(testdl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            acc(outputs, targets)
    acc_value = acc.compute().item()
    print(f"Test accuracy: {acc_value:.3f}")


@click.command()
@click.option("--train/--test", help="Whether to train or test.", default=True)
def main(train):
    if train:
        trainset, valset = BinDataset.load_train_val_single(DATAROOT)
        traindl = t.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
        )
        valdl = t.utils.data.DataLoader(
            valset, batch_size=1000, shuffle=False, pin_memory=True
        )
        net = Net().to(DEVICE)
        optim = t.optim.Adam(net.parameters(), lr=LR)
        loss_fn = t.nn.BCELoss()
        training_loop(net, optim, loss_fn, traindl, valdl)
        t.save(net, CHECKPOINT)
        print(f"Checkpoint saved at {CHECKPOINT}")
    else:
        testset = BinDataset.load_test_single(DATAROOT)
        testdl = t.utils.data.DataLoader(
            testset, batch_size=1000, shuffle=False, pin_memory=True
        )
        net = t.load(CHECKPOINT)
        evaluate(net, testdl)


if __name__ == "__main__":
    main()
