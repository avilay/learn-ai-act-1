import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
import torch as t
import torchmetrics as tm
from tqdm import tqdm

logger = logging.getLogger(__name__)
DEVICE = "empty"


class BatchMetricsAccumulator:
    def __init__(self):
        self._losses = []
        self._acc_fn = tm.Accuracy().to(DEVICE)

    def __call__(self, outputs, targets, loss):
        self._losses.append(loss)
        self._acc_fn.update(outputs, targets)

    def compute(self):
        return sum(self._losses) / len(self._losses), self._acc_fn.compute()


class MyBCELoss(t.nn.BCELoss):
    def __init__(self):
        super().__init__()

    def __call__(self, outputs, targets):
        return super().__call__(outputs, targets.to(t.float32))


class Net(t.nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        logger.info("Instantiating BinClassifer.")
        self.fc1 = t.nn.Linear(20, 8)
        self.fc2 = t.nn.Linear(8, 1)

    def forward(self, batch_x: t.Tensor) -> t.Tensor:
        x = t.nn.functional.relu(self.fc1(batch_x))
        batch_y_hat = t.sigmoid(self.fc2(x))
        return t.squeeze(batch_y_hat, dim=1)


def _to_tensor_ds(filename):
    data = pd.read_csv(filename).values
    X = t.from_numpy(data[:, :-1]).to(t.float).to(DEVICE)
    Y = t.from_numpy(data[:, -1]).to(t.int).to(DEVICE)
    dataset = t.utils.data.TensorDataset(X, Y)
    return dataset


def build_train_datasets(datapath):
    if not Path(datapath).exists():
        raise RuntimeError(f"Data not found at {datapath}!")

    trainfile = datapath / "train.csv"
    trainset = _to_tensor_ds(trainfile)

    valfile = datapath / "val.csv"
    valset = _to_tensor_ds(valfile)

    return trainset, valset


def build_test_dataset(datapath):
    testfile = datapath / "test.csv"
    testset = _to_tensor_ds(testfile)
    return testset


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


def _validate(model, loss_fn, valdl, acc_metrics=lambda x, y, z: 0):
    model.eval()
    with t.no_grad():
        for inputs, targets in valdl:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            acc_metrics(outputs, targets, loss.detach().item())


def train(trainset, valset, model, MetricsAccumulator, loss_fn, log_frequency, hparams):
    module = importlib.import_module("torch.optim")
    Optim = getattr(module, hparams.optim.class_name)
    optim = Optim(model.parameters(), **hparams.optim.args)

    traindl = t.utils.data.DataLoader(trainset, hparams.batch_size, shuffle=True)
    valdl = t.utils.data.DataLoader(valset, 1000)

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
        logger.error("Training interrupted")

    return model


def evaluate(testset, MetricsAccumulator, model):
    testdl = t.utils.data.DataLoader(testset, 1000)
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


@hydra.main(config_path="./conf", config_name="single-binclass")
def main(cfg):
    datapath = Path.home() / "mldata" / "binclass"

    global DEVICE
    DEVICE = t.device(cfg.device)

    if cfg.cmd == "train":
        logger.info("Starting single binclass training run")
        trainset, valset = build_train_datasets(datapath)
        model = Net(cfg.hparams).to(DEVICE)
        loss_fn = MyBCELoss()
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
