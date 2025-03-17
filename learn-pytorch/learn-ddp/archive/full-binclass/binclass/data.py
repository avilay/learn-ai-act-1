from pathlib import Path

import pandas as pd
import torch as t
from torch.utils.data import TensorDataset


def _to_tensor_ds(filename):
    data = pd.read_csv(filename).values
    X = t.from_numpy(data[:, :-1]).to(t.float)
    Y = t.from_numpy(data[:, -1]).to(t.int)
    dataset = TensorDataset(X, Y)
    return dataset


def build_dataset(dataroot):
    dataroot = Path.expanduser(Path(dataroot))

    trainfile = dataroot / "train.csv"
    trainset = _to_tensor_ds(trainfile)

    valfile = dataroot / "val.csv"
    valset = _to_tensor_ds(valfile)

    testfile = dataroot / "test.csv"
    testset = _to_tensor_ds(testfile)

    return trainset, valset, testset
