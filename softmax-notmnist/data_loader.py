import os.path as path
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

DATAROOT = '/Users/avilay.parekh/data/notMNIST'
# Some comment here
'''
All data methods return a 3-element tuple imgpaths, X, Y.
Where imgpaths are the images corresponding to each row in X.
As usual, each row of X -> x^{(i)} is an instance of data with
N features. Y is one-hot encoded classification matrix where y^{(i)} is the
the vector represented the one-hot encoded classification of x^{(i)}.

For the batch methods, the number of rows is the batch size.
'''


class NotMnistLoader:
    def __init__(self, batch_size=100):
        train_dir = path.join(DATAROOT, 'notMNIST_train')
        all_train_files = [path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.png')]
        all_labels = [path.basename(fname)[0] for fname in all_train_files]
        self._one_hot_encoder = LabelBinarizer()
        self._one_hot_encoder.fit(all_labels)
        self._train_files, self._val_files = train_test_split(all_train_files, test_size=0.1)

        test_dir = path.join(DATAROOT, 'notMNIST_test')
        self._test_files = [path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.png')]

        self._batch_size = batch_size

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    @staticmethod
    def _transform_img(img):
        x = np.array(img, dtype=np.float32).flatten()
        return x / 255

    @staticmethod
    def _inverse_transform_img(x):
        pixels = x * 255
        return pixels.reshape(28, 28)

    def _transform_labels(self, labels):
        return self._one_hot_encoder.transform(labels)

    def _inverse_transform_labels(self, Y):
        return self._one_hot_encoder.inverse_transform(Y)

    def _load(self, imgpaths):
        X = []
        labels = []
        for imgpath in imgpaths:
            img = Image.open(imgpath)
            img.load()
            x = self._transform_img(img)
            label = path.basename(imgpath)[0]
            X.append(x)
            labels.append(label)
        Y = self._transform_labels(labels)
        return imgpaths, X, Y

    def _all(self, files):
        return self._load(files)

    def _batches(self, files):
        m = len(files)
        for i in range(0, m, self._batch_size):
            files_batch = files[i:i + self._batch_size]
            yield self._load(files_batch)

    def show(self, n, filenames, X, Y, H=None):
        m = X.shape[0]
        ndxs = np.random.choice(np.arange(m), n)
        for j in range(n):
            x = X[ndxs[j], :]
            ximg = self._inverse_transform_img(x)
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

        actual_labels = self._inverse_transform_labels(Y)
        if H:
            predicted_labels = self._inverse_transform_labels(H)
        else:
            predicted_labels = [None] * m
        for j in range(n):
            print('Actual: {}, Predicted: {}, Sourcefile: {}'.format(
                actual_labels[ndxs[j]], predicted_labels[ndxs[j]], filenames[ndxs[j]]))

    def validation_all(self):
        return self._all(self._val_files)

    def validation_batches(self):
        return self._batches(self._val_files)

    def train_all(self):
        return self._all(self._train_files)

    def train_batches(self):
        return self._batches(self._train_files)

    def test_all(self):
        return self._all(self._test_files)

    def test_batches(self):
        return self._batches(self._test_files)
