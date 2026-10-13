from pathlib import Path

import numpy as np
import pretty_traceback
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

# from .datagen import makedata
from compare.datagen import make_datasets

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


def train():
    trainset, valset = make_datasets(DATAROOT)
    traindl = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    valdl = DataLoader(valset, batch_size=10_000, shuffle=False, drop_last=True)

    net = Net()
    net.load_state_dict(t.load("net.pth"))
    optim = t.optim.Adam(net.parameters(), lr=LR)
    loss = t.nn.BCEWithLogitsLoss()

    for epoch in range(1, N_EPOCHS + 1):
        print("\nTRAIN:")
        net.train()
        train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
        with t.enable_grad():
            for X, y in tqdm(traindl):
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
            for X, y in tqdm(valdl):
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
