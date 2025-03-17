import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

DATAROOT = '/Users/avilay.parekh/data/notMNIST'
BATCH_SIZE = 128


def listpngs(dirname):
    imgs = []
    for filename in os.listdir(dirname):
        if filename.endswith('.png'):
            imgs.append(path.join(dirname, filename))
    return imgs


class NotMnistLoader:
    def __init__(self, val_frac=0.05, flatten=False):
        self._flatten = flatten

        train_dir = path.join(DATAROOT, 'notMNIST_train')
        all_train_files = np.random.choice(listpngs(train_dir), 15000, replace=False)

        test_dir = path.join(DATAROOT, 'notMNIST_test')
        self._test_files = np.random.choice(listpngs(test_dir), 1000, replace=False)

        # num_val_samples = int(len(all_train_files) * val_frac)
        # self._val_files = np.random.choice(all_train_files, num_val_samples, replace=False)
        # self._train_files = list(set(all_train_files) - set(self._val_files))
        self._train_files, self._val_files = train_test_split(all_train_files, test_size=val_frac, random_state=832289)

        all_train_labels = [path.basename(fn)[0] for fn in self._train_files]
        self._label_encoder = LabelBinarizer()
        self._label_encoder.fit(all_train_labels)

    def _transform_img(self, x):
        x = (x - 128) / 128
        return x

    def _inverse_transform_img(self, x):
        x = (x * 128) + 128
        return x.reshape(28, 28) if self._flatten else x

    def show(self, n, filenames, X, Y, H=None):
        m = X.shape[0]
        ndxs = np.random.choice(np.arange(m), n)
        for j in range(n):
            x = X[j, :]
            ximg = self._inverse_transform_img(x)
            if self._flatten:
                ximg = ximg.reshape(28, 28)
            fig = plt.subplot(1, n, j + 1)
            fig.tick_params(
                axis='both',
                which='both',
                bottom='off',
                top='off',
                left='off',
                labelleft='off',
                labelbottom='off')
            fig.imshow(ximg, cmap='gray')

        actual_labels = self._label_encoder.inverse_transform(Y)
        if H:
            predicted_labels = self._label_encoder.inverse_transform(H)
        else:
            predicted_labels = [None] * m
        for j in range(n):
            print('Actual: {}, Predicted: {}, Sourcefile: {}'.format(
                actual_labels[j], predicted_labels[j], filenames[j]))

    def _load_imgs(self, imgpaths):
        m = len(imgpaths)
        if self._flatten:
            X = np.full((m, 28*28), fill_value=np.nan)
        else:
            X = np.full((m, 28, 28), fill_value=np.nan)
        y = [None] * m
        for i, imgpath in enumerate(imgpaths):
            features = np.copy(mpimg.imread(imgpath))
            label = path.basename(imgpath)[0]
            X[i, :] = features.flatten() if self._flatten else features
            y[i] = label
        X_norm = self._transform_img(X)
        Y_onehot = self._label_encoder.transform(y)
        return imgpaths, X_norm, Y_onehot

    def validation_all(self):
        return self._load_imgs(self._val_files)

    def validation_batches(self):
        for i in range(0, len(self._val_files), BATCH_SIZE):
            file_batch = self._val_files[i:i+BATCH_SIZE]
            yield self._load_imgs(file_batch)

    def train_all(self):
        return self._load_imgs(self._train_files)

    def train_batches(self):
        for i in range(0, len(self._train_files), BATCH_SIZE):
            file_batch = self._train_files[i:i+BATCH_SIZE]
            yield self._load_imgs(file_batch)

    def test_all(self):
        return self._load_imgs(self._test_files)

    def test_batches(self):
        for i in range(0, len(self._test_files), BATCH_SIZE):
            file_batch = self._test_files[i:i+BATCH_SIZE]
            yield self._load_imgs(file_batch)
