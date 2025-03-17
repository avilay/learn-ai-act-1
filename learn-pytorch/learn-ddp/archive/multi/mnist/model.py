import torch as t
import torch.nn.functional as F
import torchmetrics as tm


DEVICE = t.device("cuda:0" if t.cuda.is_available() else "cpu")


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
