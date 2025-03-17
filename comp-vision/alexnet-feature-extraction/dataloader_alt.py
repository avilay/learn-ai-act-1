import os.path as path
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DATAROOT = '/data/traffic-signs/'


class TrafficSignsLoader:
    def __init__(self):
        datafile = path.join(DATAROOT, 'train_alt.p')
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                data['features'], data['labels'], test_size=0.2, random_state=0)
        
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        
        self._batch_size = None

    def set_batch_size(self, batch_size):
        self._batch_size = batch_size

    def validation_all(self):
        return self.X_val, self.y_val

    def _batches(self, X, y):
        num_rows = (X.shape[0] // self._batch_size) * self._batch_size
        X = X[:num_rows]
        y = y[:num_rows]
        for i in range(0, num_rows, self._batch_size):
            X_batch = X[i:i+self._batch_size]
            y_batch = y[i:i+self._batch_size]
            yield X_batch, y_batch

    def validation_batches(self):
        X_val, y_val = self.validation_all()
        return self._batches(X_val, y_val)

    def train_all(self):
        return self.X_train, self.y_train

    def train_batches(self):
        X_train, y_train = self.train_all()
        return self._batches(X_train, y_train)
