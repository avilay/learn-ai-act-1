import logging

import torch as t
import torchmetrics as tm

DEVICE = t.device("cuda:0" if t.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


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


# class BinClassifierComplex(t.nn.Module):
#     def __init__(self):
#         super().__init__()
#         logger.info("Instantiating BinClassifierComplex.")
#         self.fc1 = t.nn.Linear(20, 200)
#         self.fc2 = t.nn.Linear(200, 200)
#         self.fc3 = t.nn.Linear(200, 100)
#         self.fc4 = t.nn.Linear(100, 50)
#         self.fc5 = t.nn.Linear(50, 25)
#         self.fc6 = t.nn.Linear(25, 12)
#         self.fc7 = t.nn.Linear(12, 1)

#     def forward(self, x):
#         x = t.nn.functional.relu(self.fc1(x))
#         x = t.nn.functional.relu(self.fc2(x))
#         x = t.nn.functional.relu(self.fc3(x))
#         x = t.nn.functional.relu(self.fc4(x))
#         x = t.nn.functional.relu(self.fc5(x))
#         x = t.nn.functional.relu(self.fc6(x))
#         y_hat = t.sigmoid(self.fc7(x))
#         return t.squeeze(y_hat, dim=1)
