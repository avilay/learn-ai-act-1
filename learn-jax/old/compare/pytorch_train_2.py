from pathlib import Path

import numpy as np
import pretty_traceback
import torch as t
from tqdm import tqdm

# from .datagen import makedata
from compare.datagen import make_ndarray

pretty_traceback.install()

# Hyperparams
LR = 0.003
N_EPOCHS = 2
BATCH_SIZE = 32

# Data config
DATAROOT = Path.home() / "mldata" / "binclass"


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = t.nn.Linear(20, 32)
        self.fc2 = t.nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = t.nn.functional.relu(x)
        logits = self.fc2(x)
        return logits.squeeze(dim=1)


def batches(X, y, batch_size):
    stop = 0
    while stop < X.shape[0]:
        start = stop
        stop = start + batch_size
        batch_X = X[start:stop]
        batch_y = y[start:stop]
        yield (batch_X, batch_y)


def train():
    device = t.device("mps")

    X_train, X_val, y_train, y_val = make_ndarray(DATAROOT)
    X_train, X_val, y_train, y_val = (
        t.tensor(X_train).to(device),
        t.tensor(X_val).to(device),
        t.tensor(y_train).to(device),
        t.tensor(y_val).to(device),
    )
    X_train.pin_memory()
    y_train.pin_memory()
    X_val.pin_memory()
    y_val.pin_memory()

    net = Net()
    # net.load_state_dict(t.load("net.pth"))
    net.to(device)
    optim = t.optim.Adam(net.parameters(), lr=LR)
    loss = t.nn.BCEWithLogitsLoss()

    for epoch in range(1, N_EPOCHS + 1):
        print("\nTRAIN:")
        net.train()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        with t.enable_grad():
            for X, y in tqdm(
                batches(X_train, y_train, BATCH_SIZE),
                total=X_train.shape[0] // BATCH_SIZE,
            ):
                X = X.to(device)
                y = y.to(device)

                optim.zero_grad()
                logits = net(X)
                train_loss = loss(logits, y.to(t.float32))
                train_loss.backward()
                optim.step()

                train_losses.append(train_loss.detach().item())

                probs = t.sigmoid(logits.detach())
                y_hat = t.where(probs > 0.5, 1, 0)
                train_accuracy = t.mean(t.where(y_hat == y, 1, 0).to(t.float32)).item()
                train_accuracies.append(train_accuracy)

        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)

        print("\nVALIDATE:")
        net.eval()
        with t.no_grad():
            for X, y in tqdm(
                batches(X_val, y_val, 10_000), total=X_val.shape[0] // 10_000
            ):
                X = X.to(device)
                y = y.to(device)

                logits = net(X)
                val_loss = loss(logits, y.to(t.float32))
                val_losses.append(val_loss.detach().item())
                probs = t.sigmoid(logits.detach())
                y_hat = t.where(probs > 0.5, 1, 0)
                val_accuracy = t.mean(t.where(y_hat == y, 1, 0).to(t.float32)).item()
                val_accuracies.append(val_accuracy)
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accuracies)
        net.train()

        print(
            f"\n{epoch}: Val Loss={avg_val_loss:.5f}, Train Loss={avg_train_loss:.5f}, Val Acc = {avg_val_acc:.3f}, Train Acc = {avg_train_acc:.3f}\n"
        )


def main():
    train()


if __name__ == "__main__":
    main()
