import os
import os.path as path

import numpy as np
import matplotlib.image as mpimg
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split


def listpngs(dirname):
    imgs = []
    for filename in os.listdir(dirname):
        if filename.endswith('.png'):
            imgs.append(path.join(dirname, filename))
    return imgs


def to_dataset(imgnames: list) -> tuple:
    """

    :param imgnames:
    :return:
    """
    X = []
    y = []
    for imgname in imgnames:
        x = np.copy(mpimg.imread(imgname)).flatten()
        x = x.astype(np.float32)
        X.append(x)
        y.append(path.basename(imgname)[0])
    X = np.array(X)
    return X, y


class NotMnistLoader:
    def __init__(self):
        all_test_imgs = listpngs('/Users/avilay.parekh/data/notMNIST/notMNIST_test')
        all_train_imgs = listpngs('/Users/avilay.parekh/data/notMNIST/notMNIST_train')

        train_val_imgnames = np.random.choice(all_train_imgs, 15000, replace=False)
        test_imgnames = np.random.choice(all_test_imgs, 1000, replace=False)

        X_train_val_raw, y_train_val = to_dataset(train_val_imgnames)

        # Scale and one-hot encode the training data
        label_encoder = LabelBinarizer().fit(y_train_val)
        self.pixel_scaler = StandardScaler().fit(X_train_val_raw)
        self._scale_pixels = self._ranged_pixel_scaler
        # self._scale_pixels = self._std_pixel_scaler

        X_train_val = self._scale_pixels(X_train_val_raw)
        Y_train_val = label_encoder.transform(y_train_val).astype(np.float32)

        X_test_raw, y_test = to_dataset(test_imgnames)

        # Use the training scaler and one-hot encoder to scale and encode test data
        self._X_test = self._scale_pixels(X_test_raw)
        self._Y_test = label_encoder.transform(y_test).astype(np.float32)

        self._X_train, self._X_val, self._Y_train, self._Y_val, train_imgnames, val_imgnames = train_test_split(
            X_train_val, Y_train_val, train_val_imgnames, test_size=0.05, random_state=832289)

    def _simple_pixel_scaler(self, img):
        return img / 255

    @property
    def _std_pixel_scaler(self, img):
        return self.pixel_scaler.transform(img)

    def _ranged_pixel_scaler(self, img):
        a = 0.1
        b = 0.9
        grayscale_min = 0
        grayscale_max = 255
        return a + (((img - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))

    def validation_all(self):
        return None, self._X_val, self._Y_val

    def validation_batches(self):
        raise RuntimeError('Not Implemented!')

    def train_all(self):
        raise RuntimeError('Not Implemented!')

    def train_batches(self):
        m = self._X_train.shape[0]
        for i in range(0, m, 100):
            batch_X_train = self._X_train[i:i + 100]
            batch_Y_train = self._Y_train[i:i + 100]
            yield None, batch_X_train, batch_Y_train

    def test_all(self):
        return None, self._X_test, self._Y_test

    def test_batches(self):
        raise RuntimeError('Not Implemented!')


# if __name__ == '__main__':
#     dl = NotMnistLoade
