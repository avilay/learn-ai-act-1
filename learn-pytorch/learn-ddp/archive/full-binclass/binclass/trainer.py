import logging
import torch as t
import importlib
import numpy as np
import torchmetrics

logger = logging.getLogger(__name__)


def train(cfg, traindl, valdl, model_cls):
    model = model_cls()
    module = importlib.import_module("torch.optim")
    optim_cls = getattr(module, cfg.optimizer)
    optim = optim_cls(model.parameters(), lr=cfg.hparams.lr)
    loss_fn = t.nn.BCELoss()
    try:
        for epoch in range(1, cfg.hparams.n_epochs + 1):
            calc_metrics = epoch % cfg.log_frequency == 0
            if calc_metrics:
                train_acc_fn = torchmetrics.Accuracy()
                train_losses = []

            model.train()
            with t.enable_grad():
                for batch_inputs, batch_targets in traindl:
                    # the standard 5-step process for tranining
                    optim.zero_grad()
                    batch_outputs = model.forward(batch_inputs)
                    loss = loss_fn(batch_outputs, batch_targets.to(t.float))
                    loss.backward()
                    optim.step()

                    if calc_metrics:
                        train_losses.append(loss.detach().item())
                        train_acc_fn.update(batch_outputs, batch_targets)

            if calc_metrics:
                train_loss = np.mean(train_losses)
                train_acc = train_acc_fn.compute()

                model.eval()
                val_losses = []
                val_acc_fn = torchmetrics.Accuracy()
                with t.no_grad():
                    for batch_inputs, batch_targets in valdl:
                        batch_outputs = model(batch_inputs)
                        loss = loss_fn(batch_outputs, batch_targets.to(t.float))
                        val_losses.append(loss.detach().item())
                        val_acc_fn.update(batch_outputs, batch_targets)
                val_loss = np.mean(val_losses)
                val_acc = val_acc_fn.compute()

                logger.info(
                    f"Epoch {epoch}: Train: loss={train_loss:.3f} acc={train_acc:.3f} | Val: loss={val_loss:.3f} acc={val_acc:.3f}"
                )
    except KeyboardInterrupt:
        logger.error("Interrupted training!")
    return model


def test(testdl, model):
    model.eval()
    acc_fn = torchmetrics.Accuracy()
    with t.no_grad():
        for batch_inputs, batch_targets in testdl:
            batch_outputs = model(batch_inputs)
            acc_fn.update(batch_outputs, batch_targets)
    acc = acc_fn.compute()
    return acc
