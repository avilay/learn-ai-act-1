import torch as t


class BinClassifier(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = t.nn.Linear(20, 8)
        self.fc2 = t.nn.Linear(8, 1)

    def forward(self, batch_x: t.Tensor) -> t.Tensor:
        x = t.nn.functional.relu(self.fc1(batch_x))
        batch_y_hat = t.sigmoid(self.fc2(x))
        return t.squeeze(batch_y_hat, dim=1)
