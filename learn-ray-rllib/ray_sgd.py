import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.examples.train_example import LinearDataset

import torch
from torch.utils.data import DataLoader


class CustomTrainingOperator(TrainingOperator):
    def setup(self, config):
        train_loader = DataLoader(LinearDataset(2, 5), config["batch_size"])
        val_loader = DataLoader(LinearDataset(2, 5), config["batch_size"])

        model = torch.nn.Linear(1, 1)
        optim = torch.optim.SGD(model.parameters(), lr=1e-2)
        loss = torch.nn.MSELoss()
        self.model, self.optimizer, self.criterion = self.register(
            models=model, optimizers=optim, criterion=loss
        )

        self.register_data(train_loader=train_loader, validation_loader=val_loader)


ray.init(address="auto")

trainer1 = TorchTrainer(
    training_operator_cls=CustomTrainingOperator,
    num_workers=2,
    use_gpu=False,
    config={"batch_size": 64},
)

stats = trainer1.train()
print(stats)
trainer1.shutdown()
print("Success!")
