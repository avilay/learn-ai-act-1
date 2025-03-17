import logging
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path

import torch as t
import torchvision as tv
from haikunator import Haikunator
from sklearn.metrics import accuracy_score
from snippets.log_config import configure_logger
from torch.utils.data import DataLoader, random_split
from torchutils import Hyperparams, Trainer, TrainerArgs, evaluate
from torchutils.ml_loggers.stdout_logger import StdoutMLExperiment

FashionMnistDatasets = namedtuple(
    "FashionMnistDatasets", ["trainset", "valset", "testset"]
)


def create_datasets(datadir, xform):
    fashion_mnist = tv.datasets.FashionMNIST(
        root=datadir, download=True, transform=xform
    )
    train_size = int(len(fashion_mnist) * 0.8)
    val_size = len(fashion_mnist) - train_size
    trainset, valset = random_split(fashion_mnist, [train_size, val_size])
    logging.info(f"Training set size: {train_size}, Validation set size: {val_size}")

    testset = tv.datasets.FashionMNIST(
        datadir, train=False, download=True, transform=xform
    )
    logging.info(f"Test set size: {len(testset)}")

    return FashionMnistDatasets(trainset=trainset, valset=valset, testset=testset)


def build_model():
    model = t.nn.Sequential(
        OrderedDict(
            [
                ("flatten", t.nn.Flatten()),
                ("fc1", t.nn.Linear(784, 128)),
                ("relu1", t.nn.ReLU()),
                ("fc2", t.nn.Linear(128, 64)),
                ("relu2", t.nn.ReLU()),
                ("fc3", t.nn.Linear(64, 32)),
                ("relu3", t.nn.ReLU()),
                ("logits", t.nn.Linear(32, 10)),
            ]
        )
    )
    return model


@dataclass
class MyHyperparams(Hyperparams):
    batch_size: int
    n_epochs: int
    lr: float


def accuracy(y_true, y_hat):
    y_pred = t.argmax(y_hat, dim=1)
    return accuracy_score(y_true, y_pred)


def build_trainer(hparams, trainset, valset):
    run_name = Haikunator().haikunate()
    logging.info(f"Starting run {run_name}")
    model = build_model()
    optim = t.optim.Adam(model.parameters(), lr=hparams.lr)
    loss_fn = t.nn.CrossEntropyLoss()
    traindl = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True)
    valdl = DataLoader(valset, batch_size=1000)
    return TrainerArgs(
        run_name=run_name,
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        trainloader=traindl,
        valloader=valdl,
        n_epochs=hparams.n_epochs,
    )


def main():
    configure_logger()
    datadir = Path.home() / "mldata" / "fashion-mnist"
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,), (0.5,))]
    )
    fashion_mnist = create_datasets(datadir, xform)

    exp = StdoutMLExperiment("fashion-mnist-exp")

    hparams = MyHyperparams(batch_size=16, n_epochs=10, lr=0.003)
    trainer = Trainer(
        exp, fashion_mnist.trainset, fashion_mnist.valset, metric_functions=[accuracy]
    )
    trainer.metrics_log_frequency = 1
    trainer.model_log_frequency = 5
    trainer.train(hparams, build_trainer)

    testdl = DataLoader(fashion_mnist.testset, batch_size=5000)
    test_metrics = evaluate(trainer.model, testdl, [accuracy])
    test_accuracy = test_metrics["accuracy"]
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
