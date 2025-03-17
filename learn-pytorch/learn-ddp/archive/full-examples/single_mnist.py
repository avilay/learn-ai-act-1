import importlib
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import hydra
import torch as t
import torch.nn.functional as F
import torchmetrics as tm
import torchvision as tv
from tqdm import tqdm

logger = logging.getLogger(__name__)
DEVICE = "empty"


class BatchMetricsAccumulator:
    def __init__(self):
        self._losses = []
        self._acc_fn = tm.Accuracy().to(DEVICE)

    def __call__(self, outputs, targets, loss):
        self._losses.append(loss)
        preds = t.argmax(outputs, dim=1)
        self._acc_fn.update(preds, targets)

    def compute(self):
        return sum(self._losses) / len(self._losses), self._acc_fn.compute()


class Net(t.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = t.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = t.nn.Dropout2d(hparams.dropouts[0])
        self.dropout2 = t.nn.Dropout2d(hparams.dropouts[1])
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


def _validate(model, loss_fn, valdl, acc_metrics=lambda x, y, z: 0):
    model.eval()
    with t.no_grad():
        for inputs, targets in valdl:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            acc_metrics(outputs, targets, loss.detach().item())


def _train(model, loss_fn, optim, traindl, acc_metrics=lambda x, y, z: 0):
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

            acc_metrics(outputs, targets, loss.detach().item())


def train(trainset, valset, model, MetricsAccumulator, loss_fn, log_frequency, hparams):
    module = importlib.import_module("torch.optim")
    Optim = getattr(module, hparams.optim.class_name)
    optim = Optim(model.parameters(), **hparams.optim.args)

    traindl = t.utils.data.DataLoader(trainset, hparams.batch_size, shuffle=True)
    valdl = t.utils.data.DataLoader(valset, min(1000, len(valset)))

    logger.info(f"Starting training on {DEVICE}")

    try:
        for epoch in range(1, hparams.epochs + 1):
            if epoch % log_frequency == 0:
                acc_train_batch_metrics = MetricsAccumulator()
                _train(model, loss_fn, optim, traindl, acc_train_batch_metrics)
                train_loss, train_acc = acc_train_batch_metrics.compute()

                acc_val_batch_metrics = MetricsAccumulator()
                _validate(model, loss_fn, valdl, acc_val_batch_metrics)
                val_loss, val_acc = acc_val_batch_metrics.compute()

                logger.info(
                    f"Epoch {epoch}: Train [loss={train_loss:.3f} acc={train_acc:.3f}] | Val [loss={val_loss:.3f} acc={val_acc:.3f}]"
                )
                print("\n")
            else:
                _train(model, loss_fn, optim, traindl)
    except KeyboardInterrupt:
        logger.error("Training interrupted!")

    return model


def evaluate(testset, MetricsAccumulator, model):
    testdl = t.utils.data.DataLoader(testset, min(1000, len(testset)))
    model.eval()
    metrics = MetricsAccumulator()
    with t.no_grad():
        for inputs, targets in tqdm(testdl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            metrics(outputs, targets, 0.0)
    _, acc = metrics.compute()
    return acc


def _build_xform():
    means = [0.5]
    stds = [0.5]
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize(means, stds)]
    )
    return xform


def build_train_datasets(datapath):
    xform = _build_xform()
    traindata = tv.datasets.MNIST(datapath, download=True, train=True, transform=xform)
    trainsize = int(0.9 * len(traindata))
    valsize = len(traindata) - trainsize
    trainset, valset = t.utils.data.random_split(traindata, (trainsize, valsize))
    logger.info(
        f"Prepared trainset with {len(trainset)} examples and valset with {len(valset)} examples."
    )
    return trainset, valset


def build_test_dataset(datapath):
    xform = _build_xform()
    testset = tv.datasets.MNIST(datapath, download=True, train=False, transform=xform)
    logger.info(f"Prepared testset with {len(testset)} examples.")
    return testset


@hydra.main(config_path="./conf", config_name="single-mnist")
def main(cfg):
    datapath = Path.home() / "mldata" / "mnist"

    global DEVICE
    DEVICE = t.device(cfg.device)

    if cfg.cmd == "train":
        logger.info("Starting single mnist training run")
        trainset, valset = build_train_datasets(datapath)
        model = Net(cfg.hparams).to(DEVICE)
        loss_fn = t.nn.NLLLoss()
        start = datetime.now()
        train(
            trainset,
            valset,
            model,
            BatchMetricsAccumulator,
            loss_fn,
            cfg.log_frequency,
            cfg.hparams,
        )
        end = datetime.now()
        logger.info(f"Training took {end - start}")
        checkpoint = "model.ckpt"
        logger.info(f"Saving model to {os.getcwd()}/{checkpoint}")
        t.save(model, checkpoint)
    elif cfg.cmd == "test":
        testset = build_test_dataset(datapath)
        checkpoint = Path(cfg.checkpoint)
        model = t.load(checkpoint)
        accuracy = evaluate(testset, BatchMetricsAccumulator, model)
        print(f"Test accuracy: {accuracy:.3f}")
    else:
        print("Unknown command")
        sys.exit(1)


if __name__ == "__main__":
    main()
