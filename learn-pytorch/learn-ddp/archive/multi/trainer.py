import importlib
import logging

import torch as t
from tqdm import tqdm


DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def _validate(model, loss_fn, valdl, acc_metrics=lambda x, y, z: 0):
    model.eval()
    with t.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valdl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            acc_metrics(outputs, targets, loss.detach().item())

            logger.debug(f"Processed {batch_idx} of valset.")


def _train(model, loss_fn, optim, traindl, acc_metrics=lambda x, y, z: 0):
    model.train()
    with t.enable_grad():
        for batch_idx, (inputs, targets) in enumerate(traindl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # The standard 5-step training process
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()

            acc_metrics(outputs, targets, loss.detach().item())
            if batch_idx % 10 == 0:
                logger.debug(f"Processed {batch_idx}")


def train(traindl, valdl, model, MetricsAccumulator, loss_fn, log_frequency, hparams):

    module = importlib.import_module("torch.optim")
    Optim = getattr(module, hparams.optim.class_name)
    optim = Optim(model.parameters(), **hparams.optim.args)

    logger.info(f"Starting training on {DEVICE}")

    try:
        for epoch in range(1, hparams.epochs + 1):
            if epoch % log_frequency == 0:
                acc_train_batch_metrics = MetricsAccumulator()
                _train(model, loss_fn, optim, traindl, acc_train_batch_metrics)
                train_loss, train_acc = acc_train_batch_metrics.compute()

                logger.info(
                    f"Epoch {epoch}: Train [loss={train_loss:.3f} acc={train_acc:.3f}]"
                )

                acc_val_batch_metrics = MetricsAccumulator()
                _validate(model, loss_fn, valdl, acc_val_batch_metrics)
                logger.debug("Finished validating model. Computing metrics.")
                val_loss, val_acc = acc_val_batch_metrics.compute()
                logger.debug("Finished computing validation metrics.")

                logger.info(
                    f"Epoch {epoch}: Val [loss={val_loss:.3f} acc={val_acc:.3f}]"
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
