import os
import os.path as path
import pickle
import numpy as np
from .consts import *


def split():
    if not path.exists(CIFAR_10_ROOT):
        raise RuntimeError('Source data does not exist!')
    if path.exists(CIFAR_100_ROOT) and len(os.listdir(CIFAR_100_ROOT)):
        raise RuntimeError('Data has already been split!')

    if not path.exists(CIFAR_100_ROOT):
        os.mkdir(CIFAR_100_ROOT)

    val = dict(
        labels=[],
        filenames=[],
        data=np.full((5000, 3072), fill_value=np.nan)
    )

    batch_ctr = 1
    val_ctr = 0
    for batch_num in range(1, 6):
        old_batch_file = 'data_batch_{}'.format(batch_num)
        old_batch_filepath = path.join(CIFAR_10_ROOT, old_batch_file)
        with open(old_batch_filepath, 'rb') as f:
            old_batch = pickle.load(f, encoding='latin1')
        for i in range(0, 9000, 1000):
            new_batch = dict(
                labels=old_batch['labels'][i:i+1000],
                filenames=old_batch['filenames'][i:i+1000],
                data=old_batch['data'][i:i+1000, :]
            )
            new_batch_file = 'data_batch_{}'.format(batch_ctr)
            batch_ctr += 1
            new_batch_filepath = path.join(CIFAR_100_ROOT, new_batch_file)
            with open(new_batch_filepath, 'wb') as f:
                pickle.dump(new_batch, f, pickle.HIGHEST_PROTOCOL)

        val['labels'] += old_batch['labels'][9000:]
        val['filenames'] += old_batch['filenames'][9000:]
        val['data'][val_ctr:val_ctr+1000, :] = old_batch['data'][9000:, :]
        val_ctr += 1000

    val_file = 'data_val'
    val_filepath = path.join(CIFAR_100_ROOT, val_file)
    with open(val_filepath, 'wb') as f:
        pickle.dump(val, f, pickle.HIGHEST_PROTOCOL)

    old_test_file = 'test_batch'
    old_test_filepath = path.join(CIFAR_10_ROOT, old_test_file)
    with open(old_test_filepath, 'rb') as f:
        old_test_batch = pickle.load(f, encoding='latin1')
    test_batch_ctr = 1
    for i in range(0, 10000, 1000):
        new_test_batch = dict(
            labels=old_test_batch['labels'][i:i+1000],
            filenames=old_test_batch['filenames'][i:i+1000],
            data=old_test_batch['data'][i:i+1000, :]
        )
        new_test_batch_file = 'test_batch_{}'.format(test_batch_ctr)
        test_batch_ctr += 1
        new_test_batch_filepath = path.join(CIFAR_100_ROOT, new_test_batch_file)
        with open(new_test_batch_filepath, 'wb') as f:
            pickle.dump(new_test_batch, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    split()
