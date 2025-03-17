"""
Trainset has 54,016 instances (422 batches with 128 instances)
Testset has 10,000 instances (10 batches with 1000 instances)

On GCP
  * With CPU:
    - 1:09 mins per epoch
    - At the end of 3 epochs
      - Train: loss=0.314 acc=0.908
      - Val: loss=0.164 acc=0.954
      - Test accuracy: 0.944
  * With GPU:
    - 23 seconds per epoch
    - At the end of 3 epochs
      - Train: loss=0.365 acc=0.892
      - Val: loss=0.181 acc=0.949
      - Test accuracy: 0.938
"""
from pathlib import Path
import click
import torch as t
import torchmetrics as tm
import torchvision as tv
from tqdm import tqdm
import torch.nn.functional as F


# Hyperparams
N_EPOCHS = 7
BATCH_SIZE = 128
DROPOUTS = [0.25, 0.5]
MOMENTUM = 0.9
LR = 0.001

# DEVICE = t.device("cpu")
DEVICE = t.device("cuda:0")
DATAROOT = Path.home() / "mldata" / "mnist"
CHECKPOINT = Path.home() / "mlruns" / "single_mnist_net.ckpt"


class Net(t.nn.Module):
    def __init__(self, dropouts):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout2d(dropouts[0])
        self.dropout2 = t.nn.Dropout2d(dropouts[1])
        self.fc1 = t.nn.Linear(9216, 128)
        self.fc2 = t.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = t.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def build_train_datasets(dataroot):
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
    )
    mnist = tv.datasets.MNIST(dataroot, train=True, transform=xform)
    train_size = int(len(mnist) * 0.9)
    trainset = t.utils.data.Subset(mnist, range(train_size))
    valset = t.utils.data.Subset(mnist, range(train_size, len(mnist)))
    return trainset, valset


def build_test_datasets(dataroot):
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize([0.5], [0.5])]
    )
    testset = tv.datasets.MNIST(dataroot, train=False, transform=xform)
    return testset


def training_loop(model, optim, loss_fn, traindl, valdl):
    print(f"Using device {DEVICE}")
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
                loss = loss_fn(outputs, targets)
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
                loss = loss_fn(outputs, targets)

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
@click.option("--train/--test", help="Whether to train or test.", required=True)
def main(train):
    if train:
        trainset, valset = build_train_datasets(DATAROOT)
        traindl = t.utils.data.DataLoader(
            trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
        )
        valdl = t.utils.data.DataLoader(
            valset, batch_size=1000, shuffle=False, pin_memory=True
        )
        net = Net(DROPOUTS).to(DEVICE)
        optim = t.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
        loss_fn = t.nn.NLLLoss()
        training_loop(net, optim, loss_fn, traindl, valdl)
        t.save(net, CHECKPOINT)
        print(f"Checkpoint saved at {CHECKPOINT}")
    else:
        testset = build_test_datasets(DATAROOT)
        testdl = t.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
        net = t.load(CHECKPOINT)
        evaluate(net, testdl)


if __name__ == "__main__":
    main()
