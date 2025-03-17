from collections import namedtuple
from datetime import datetime

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

CONF = {
    'image_shape': (28, 28, 1),
    'filters': 32,
    'pool_size': 2,
    'kernel_size': 3,
    'num_classes': 5    
}
HyperParams = namedtuple('HyperParams', ['batch_size', 'epochs'])


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train_lt5 = X_train[y_train < 5]
    y_train_lt5 = y_train[y_train < 5]
    X_test_lt5 = X_test[y_test < 5]
    y_test_lt5 = y_test[y_test < 5]

    X_train_gte5 = X_train[y_train >= 5]
    y_train_gte5 = y_train[y_train >= 5] - 5
    X_test_gte5 = X_test[y_test >= 5]
    y_test_gte5 = y_test[y_test >= 5] - 5

    data = {
        'lt5': {
            'train': (X_train_lt5, y_train_lt5),
            'test': (X_test_lt5, y_test_lt5)
        },
        'gte5': {
            'train': (X_train_gte5, y_train_gte5),
            'test': (X_test_gte5, y_test_gte5)
        }
    }
    return data


def train(model, data, hparams):
    # Pre-process data
    X_train, y_train = data['train']
    X_test, y_test = data['test']
    
    X_train = X_train.reshape((X_train.shape[0],) + CONF['image_shape']).astype(np.float32)
    X_test = X_test.reshape((X_test.shape[0],) + CONF['image_shape']).astype(np.float32)
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    Y_train = keras.utils.to_categorical(y_train, CONF['num_classes'])
    Y_test = keras.utils.to_categorical(y_test, CONF['num_classes'])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    start = datetime.now()
    model.fit(
        X_train, Y_train,
        batch_size=hparams.batch_size,
        epochs=hparams.epochs,
        validation_data=(X_test, Y_test)
    )
    end = datetime.now()
    print('Training time: {}'.format(end - start))
    score = model.evaluate(X_test, Y_test)
    print('Test score: {:.3f}'.format(score[0]))
    print('Test accuracy: {:.3f}'.format(score[1]))


def main():
    hparams = HyperParams(batch_size=128, epochs=5)
    data = load_data()

    feature_layers = [
        Conv2D(CONF['filters'], CONF['kernel_size'], padding='valid', input_shape=CONF['image_shape'], activation='relu'),
        Conv2D(CONF['filters'], CONF['kernel_size'], activation='relu'),
        MaxPooling2D(pool_size=CONF['pool_size']),
        Dropout(0.25),
        Flatten()
    ]

    classification_layers = [
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(CONF['num_classes'], activation='softmax')
    ]

    model = Sequential(feature_layers + classification_layers)

    # Train model for 5-digit classification [0...4]
    train(model, data['lt5'], hparams)

    # Freeze features layers and rebuild model
    for layer in feature_layers:
        layer.trainable = False
    train(model, data['gte5'], hparams)


if __name__ == '__main__':
    main()
