import matplotlib.pyplot as plt
from data_loader_old import NotMnistLoader


def main():
    dl = NotMnistLoader()
    val_files = set(dl._val_files)
    train_files = set(dl._train_files)
    test_files = set(dl._test_files)

    num_vals = len(dl._val_files)
    num_train = len(dl._train_files)
    num_test = len(dl._test_files)

    print('# val files: {}, # train files: {}, # test files: {}'.format(
        num_vals, num_train, num_test
    ))

    assert len(val_files) == num_vals, 'distinct files: {}, total files: {}'.format(len(val_files), num_vals)
    assert len(train_files) == num_train, 'distinct files: {}, total files: {}'.format(len(train_files), num_train)
    assert len(test_files) == num_test, 'distinct files: {}, total files: {}'.format(len(test_files), num_test)

    common_files = train_files.intersection(val_files)
    assert not common_files, '{} duplicate entries found'.format(len(common_files))

    print('Sample training data')
    for filenames, X_train, Y_train in dl.train_batches():
        dl.show(5, filenames, X_train, Y_train)
        break
    plt.show()

    print('\n\nSample validation data')
    for filenames, X_val, Y_val in dl.validation_batches():
        dl.show(5, filenames, X_val, Y_val)
        break
    plt.show()

    print('\n\nSample test data')
    for filenames, X_test, Y_test in dl.test_batches():
        dl.show(5, filenames, X_test, Y_test)
        break
    plt.show()


if __name__ == '__main__':
    main()