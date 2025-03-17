import pickle
from collections import namedtuple

import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

Batch = namedtuple("Batch", ["X", "y"])


class BinDataset(Dataset):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]

    def __len__(self):
        return self._X.shape[0]


def collate(samples):
    xs, ys = zip(*samples)
    X = jnp.vstack(xs)
    # y = jnp.array([y for y in ys])
    y = jnp.hstack(ys)
    return Batch(X, y)


def make_ndarray(dataroot):
    datafile = dataroot / "data.pkl"
    if datafile.exists():
        with open(datafile, "rb") as f:
            X_train, X_val, y_train, y_val = pickle.load(f)
    else:
        X, y = make_classification(
            n_classes=2,
            n_samples=1_000_000,
            random_state=0,
            n_features=20,
            n_informative=10,
            n_redundant=7,
            n_repeated=3,
            flip_y=0.05,
            class_sep=0.5,
        )
        X = X.astype(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
        dataroot.mkdir(parents=True, exist_ok=True)
        with open(datafile, "wb") as f:
            pickle.dump((X_train, X_val, y_train, y_val), f)
    return X_train, X_val, y_train, y_val


def make_datasets(dataroot, jaxify=False):
    # Using jaxify actually makes Jax impl slower!
    X_train, X_val, y_train, y_val = make_ndarray(dataroot)
    if jaxify:
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)
    trainset = BinDataset(X_train, y_train)
    valset = BinDataset(X_val, y_val)
    return trainset, valset
