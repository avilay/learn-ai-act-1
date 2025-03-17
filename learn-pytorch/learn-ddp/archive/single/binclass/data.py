import logging
from pathlib import Path

import pandas as pd
import torch as t
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")


def _to_tensor_ds(filename):
    data = pd.read_csv(filename).values
    X = t.from_numpy(data[:, :-1]).to(t.float).to(DEVICE)
    Y = t.from_numpy(data[:, -1]).to(t.int).to(DEVICE)
    dataset = TensorDataset(X, Y)
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
