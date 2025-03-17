from torch.utils.data import TensorDataset
import sklearn.datasets as skdata
import torch as t


def binary_classification(n_samples):
    X, y = skdata.make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=7,
        n_repeated=3,
        n_classes=2,
        flip_y=0.05,  # larger values make the task hard
        class_sep=0.5,  # larger values makes the task easy
        random_state=10,
    )

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.2)

    train_X = X[:train_size]
    train_y = y[:train_size]
    trainset = TensorDataset(
        t.from_numpy(train_X).to(t.float32), t.from_numpy(train_y).to(t.float32)
    )

    val_X = X[train_size : train_size + val_size]
    val_y = y[train_size : train_size + val_size]
    valset = TensorDataset(t.from_numpy(val_X).to(t.float32), t.from_numpy(val_y).to(t.float32))

    test_X = X[train_size + val_size :]
    test_y = y[train_size + val_size :]
    testset = TensorDataset(t.from_numpy(test_X).to(t.float32), t.from_numpy(test_y).to(t.float32))

    return trainset, valset, testset


def multiclass_classification(n_samples):
    X, y = skdata.make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=7,
        n_repeated=3,
        n_classes=5,
        flip_y=0.05,  # larger values make the task hard
        class_sep=0.8,  # larger values makes the task easy
        random_state=10,
    )

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.2)

    train_X = X[:train_size]
    train_y = y[:train_size]
    trainset = TensorDataset(t.from_numpy(train_X).to(t.float32), t.from_numpy(train_y).to(t.int64))

    val_X = X[train_size : train_size + val_size]
    val_y = y[train_size : train_size + val_size]
    valset = TensorDataset(t.from_numpy(val_X).to(t.float32), t.from_numpy(val_y).to(t.int64))

    test_X = X[train_size + val_size :]
    test_y = y[train_size + val_size :]
    testset = TensorDataset(t.from_numpy(test_X).to(t.float32), t.from_numpy(test_y).to(t.int64))

    return trainset, valset, testset


def regression(n_samples):
    all_X, all_y = skdata.make_regression(n_samples=n_samples, n_features=5, noise=0.5)

    train_size = int(n_samples * 0.7)
    val_size = int(n_samples * 0.2)

    train_X = all_X[:train_size]
    train_y = all_y[:train_size]
    trainset = TensorDataset(
        t.from_numpy(train_X).to(t.float32), t.from_numpy(train_y).to(t.float32)
    )

    val_X = all_X[train_size : train_size + val_size]
    val_y = all_y[train_size : train_size + val_size]
    valset = TensorDataset(t.from_numpy(val_X).to(t.float32), t.from_numpy(val_y).to(t.float32))

    test_X = all_X[train_size + val_size :]
    test_y = all_y[train_size + val_size :]
    testset = TensorDataset(t.from_numpy(test_X).to(t.float32), t.from_numpy(test_y).to(t.float32))

    return trainset, valset, testset