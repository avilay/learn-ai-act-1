import importlib
import logging

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = t.device("cuda:0" if t.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


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


def train(trainset, valset, Net, MetricsAccumulator, loss_fn, log_frequency, hparams):
    model = Net(hparams).to(DEVICE)

    module = importlib.import_module("torch.optim")
    Optim = getattr(module, hparams.optim.class_name)
    optim = Optim(model.parameters(), **hparams.optim.args)

    traindl = DataLoader(trainset, hparams.batch_size, shuffle=True)
    valdl = DataLoader(valset, min(1000, len(valset)))

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
    testdl = DataLoader(testset, min(1000, len(testset)))
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
