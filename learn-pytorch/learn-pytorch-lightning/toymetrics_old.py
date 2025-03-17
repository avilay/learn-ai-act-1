import torch as t
from torchmetrics import Accuracy
import numpy as np
import pickle
import click
import pretty_traceback

pretty_traceback.install()

N_FEATURES = 5
BATCH_SIZE = 32
TRAIN_BATCHES_PER_EPOCH = 3
VAL_BATCHES_PER_EPOCH = 2
N_EPOCHS = 2
TRAIN_ACCS = np.array([[0.21, 0.32, 0.43], [0.45, 0.78, 0.8]])
VAL_ACCS = np.array([[0.33, 0.32], [0.63, 0.74]])

np.set_printoptions(edgeitems=30, linewidth=100000, precision=3, suppress=True)
rng = np.random.default_rng()


class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = t.nn.Linear(N_FEATURES, 1)

    def forward(self, inputs):
        z = self.fc(inputs)
        probs = t.sigmoid(z)
        return t.squeeze(probs, dim=1)


def gen_targets(desired_acc, probs):
    y_hat = (probs > 0.5).to(t.int)
    y_hat_ = 1 - y_hat
    n_correct = int(np.ceil(y_hat.shape[0] * desired_acc))
    return t.cat((y_hat[:n_correct], y_hat_[n_correct:]))


def genrun():
    loss_fn = t.nn.BCELoss()
    train_accuracy = Accuracy()
    val_accuracy = Accuracy()
    net = Net()
    with open("state_dict.pkl", "wb") as f:
        pickle.dump(net.state_dict(), f)

    optim = t.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(N_EPOCHS):
        train_accuracy.reset()
        val_accuracy.reset()

        train_accs = []
        val_accs = []

        print("\nTRAINING:")
        for batch_idx in range(TRAIN_BATCHES_PER_EPOCH):
            print(net.state_dict())

            X = rng.random((BATCH_SIZE, N_FEATURES)).astype(np.float32)
            with open(f"X_train_{epoch}_{batch_idx}.pkl", "wb") as f:
                pickle.dump(X, f)

            inputs = t.from_numpy(X)
            optim.zero_grad()
            outputs = net(inputs)

            desired_acc = TRAIN_ACCS[epoch, batch_idx]
            targets = gen_targets(desired_acc, outputs)

            y = targets.numpy()
            with open(f"y_train_{epoch}_{batch_idx}.pkl", "wb") as f:
                pickle.dump(y, f)

            loss = loss_fn(outputs, targets.to(t.float32))
            loss.backward()
            optim.step()
            train_accuracy(outputs, targets)

            # Calculate the accuracy by hand
            y_hat = (outputs.detach().numpy() > 0.5).astype(int)
            acc = np.mean((y_hat == y).astype(int))
            print(f"desired accuracy: {desired_acc}, calculated accuracy: {acc:.3f}")
            train_accs.append(acc)

        print("\nVALIDATION:")
        for batch_idx in range(VAL_BATCHES_PER_EPOCH):
            X = rng.random((BATCH_SIZE, N_FEATURES)).astype(np.float32)
            with open(f"X_val_{epoch}_{batch_idx}.pkl", "wb") as f:
                pickle.dump(X, f)

            inputs = t.from_numpy(X)
            outputs = net(inputs)
            desired_acc = VAL_ACCS[epoch, batch_idx]
            targets = gen_targets(desired_acc, outputs)
            y = targets.numpy()
            with open(f"y_val_{epoch}_{batch_idx}.pkl", "wb") as f:
                pickle.dump(y, f)
            val_accuracy(outputs, targets)

            # Calculate the accuracy by hand
            y_hat = (outputs.detach().numpy() > 0.5).astype(int)
            acc = np.mean((y_hat == y).astype(int))
            print(f"desired accuracy: {desired_acc}, calculated accuracy: {acc:.3f}")
            val_accs.append(acc)

        print(
            f"\nManual: train accuracy={np.mean(train_accs)}, val accuracy={np.mean(val_accs)}"
        )
        print(
            f"Torchmetrics: train accuracy={train_accuracy.compute().item()}, val accuracy={val_accuracy.compute().item()}"
        )


def dorun():
    pass


@click.command()
@click.option("--gen/--run", default=True)
def main(gen):
    if gen:
        genrun()
    else:
        dorun()


if __name__ == "__main__":
    main()
