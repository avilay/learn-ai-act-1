"""Functions to generate synthetic datasets

These functions wrap scikit-learn data generation APIs inside PyTorch Datasets.
"""
from torch.utils.data import TensorDataset
import sklearn.datasets as skdata
import torch as t


def binary_classification(
    n_samples: int,
    n_features=20,
    n_informative=10,
    n_redundant=7,
    n_repeated=3,
    flip_y=0.05,
    class_sep=0.5,
    random_state=10,
):
    """Generate datasets for binary classification

    Will generate three datasets for training, validation, and testing with a 70%, 20%, and 10% split.
    Each dataset will have a 2D matrix $X$ with the input data, and 1D vector $y$ with the targets, each target being
    either $0$ or $1$. For documentation on the input args and how they are used please refer to
    [sckit-learn `make_classification` documentation][https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html]
    """
    X, y = skdata.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=2,
        flip_y=flip_y,  # larger values make the task hard
        class_sep=class_sep,  # larger values makes the task easy
        random_state=random_state,
    )

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.2)

    train_X = X[:train_size]
    train_y = y[:train_size]
    trainset = TensorDataset(
        t.from_numpy(train_X).to(t.float32), t.from_numpy(train_y).to(t.int)
    )

    val_X = X[train_size : train_size + val_size]
    val_y = y[train_size : train_size + val_size]
    valset = TensorDataset(
        t.from_numpy(val_X).to(t.float32), t.from_numpy(val_y).to(t.int)
    )

    test_X = X[train_size + val_size :]
    test_y = y[train_size + val_size :]
    testset = TensorDataset(
        t.from_numpy(test_X).to(t.float32), t.from_numpy(test_y).to(t.int)
    )

    return trainset, valset, testset


def multiclass_classification(
    n_samples: int,
    n_features=20,
    n_informative=10,
    n_redundant=7,
    n_repeated=3,
    n_classes=5,
    flip_y=0.05,
    class_sep=0.8,
    random_state=10,
):
    """Generates datasets for multiclass classification

    Will generate three datasets for training, validation, and testing with a 70%, 20%, and 10% split. Each dataset
    will have a 2D matrix $X$ with the input data, and 1D vector $y$ with the targets, with each target being
    between $\left[0, k\right]$. $k$ being the number of classes. For documentation on the input args and how they are
    used please refer to
    [sckit-learn `make_classification` documentation][https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html]

    """
    X, y = skdata.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        flip_y=flip_y,  # larger values make the task hard
        class_sep=class_sep,  # larger values makes the task easy
        random_state=random_state,
    )

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.2)

    train_X = X[:train_size]
    train_y = y[:train_size]
    trainset = TensorDataset(
        t.from_numpy(train_X).to(t.float32), t.from_numpy(train_y).to(t.int64)
    )

    val_X = X[train_size : train_size + val_size]
    val_y = y[train_size : train_size + val_size]
    valset = TensorDataset(
        t.from_numpy(val_X).to(t.float32), t.from_numpy(val_y).to(t.int64)
    )

    test_X = X[train_size + val_size :]
    test_y = y[train_size + val_size :]
    testset = TensorDataset(
        t.from_numpy(test_X).to(t.float32), t.from_numpy(test_y).to(t.int64)
    )

    return trainset, valset, testset


def regression(n_samples: int, n_features=5, noise=0.5):
    """Generates datasets for regrssion

    Will generate three datasets for training, validation, and testing with a 70%, 20%, and 10% split. Each dataset
    will have a 2D matrix $X$ with the input data, and 1D vector $y$ with the targets, each target being a float32
    value. For documentation on the input args and how they are used please refer to
    [sckit-learn `make_regression` documentation][https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html]

    """
    all_X, all_y = skdata.make_regression(
        n_samples=n_samples, n_features=n_features, noise=noise
    )

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.2)

    train_X = all_X[:train_size]
    train_y = all_y[:train_size]
    trainset = TensorDataset(
        t.from_numpy(train_X).to(t.float32), t.from_numpy(train_y).to(t.float32)
    )

    val_X = all_X[train_size : train_size + val_size]
    val_y = all_y[train_size : train_size + val_size]
    valset = TensorDataset(
        t.from_numpy(val_X).to(t.float32), t.from_numpy(val_y).to(t.float32)
    )

    test_X = all_X[train_size + val_size :]
    test_y = all_y[train_size + val_size :]
    testset = TensorDataset(
        t.from_numpy(test_X).to(t.float32), t.from_numpy(test_y).to(t.float32)
    )

    return trainset, valset, testset
