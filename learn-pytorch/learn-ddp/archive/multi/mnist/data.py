import logging

import torch.utils.data as td
import torchvision as tv


logger = logging.getLogger(__name__)


def _build_xform():
    means = [0.5]
    stds = [0.5]
    xform = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize(means, stds)]
    )
    return xform


def build_train_datasets(datapath):
    xform = _build_xform()
    traindata = tv.datasets.MNIST(datapath, download=True, train=True, transform=xform)
    trainsize = int(0.9 * len(traindata))
    valsize = len(traindata) - trainsize
    trainset, valset = td.random_split(traindata, (trainsize, valsize))
    logger.info(
        f"Prepared trainset with {len(trainset)} examples and valset with {len(valset)} examples."
    )
    return trainset, valset


def build_test_dataset(datapath):
    xform = _build_xform()
    testset = tv.datasets.MNIST(datapath, download=True, train=False, transform=xform)
    logger.info(f"Prepared testset with {len(testset)} examples.")
    return testset
