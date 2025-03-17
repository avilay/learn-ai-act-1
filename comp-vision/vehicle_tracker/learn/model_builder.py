import pickle
import os.path as path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from ..data import loader
from ..features.extractor import extract_img_features
from .metrics import Metrics
from ..features.feature_spec import HogSpec, HistogramSpec, SpatialSpec, FeatureSpec, Channel

DATAROOT = '/data/vehicle-tracking'


def load_training_data(spec):
    xvecs = []
    y = []
    for img, label in loader.train_data():
        xvec = extract_img_features(img, spec)
        xvecs.append(xvec)
        if label == 'yes':
            y.append(1)
        else:
            y.append(0)
    X_train = np.array(xvecs)
    y_train = np.array(y)
    return X_train, y_train


def load_validation_data(spec):
    xvecs = []
    y = []
    for img, label in loader.val_data():
        xvec = extract_img_features(img, spec)
        xvecs.append(xvec)
        if label == 'yes':
            y.append(1)
        else:
            y.append(0)
    X_train = np.array(xvecs)
    y_train = np.array(y)
    return X_train, y_train


def train(hyperparams, save=False):
    spec = hyperparams['spec']
    C = hyperparams['C']

    X_train, y_train = load_training_data(spec)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_norm = scaler.transform(X_train)

    X_val, y_val = load_validation_data(spec)
    X_val_norm = scaler.transform(X_val)

    svm = LinearSVC(C=C)
    svm.fit(X_train_norm, y_train)

    if save:
        model = {
            'ver': '1.0.1',
            'svm': svm,
            'scaler': scaler
        }

        with open(path.join(DATAROOT, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

    metrics = Metrics()
    metrics.train_accuracy = svm.score(X_train_norm, y_train)
    metrics.train_f1 = f1_score(y_train, svm.predict(X_train_norm))
    metrics.val_accuracy = svm.score(X_val_norm, y_val)
    metrics.val_f1 = f1_score(y_val, svm.predict(X_val_norm))
    print(spec)
    print(metrics)
    return metrics


def main():
    # gold spec
    hog_spec = HogSpec(
        channel=Channel.ALL,
        orientations=11,
        pixels_per_cell=16,
        cells_per_block=2
    )
    hist_spec = HistogramSpec(channel=Channel.SECOND, bins=32)
    spatial_spec = SpatialSpec(size=32)
    spec = FeatureSpec(color_space='YCrCb', hog_spec=hog_spec, hist_spec=hist_spec, spatial_spec=spatial_spec)

    hyperparams = {
        'spec': spec,
        'C': 10
    }
    train(hyperparams, save=True)


if __name__ == '__main__':
    main()

