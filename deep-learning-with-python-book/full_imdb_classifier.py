"""
Typical invocation:
$ python full_imdb_classifier.py train --plot --extembed
"""
import argparse
import os
import os.path as path
import pickle

import matplotlib.pyplot as plt
import numpy as np

import utils
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

hyperparams = {}

MODEL_FILE = '/data/learn-keras/full_imdb_classifier_model.h5'
TOK_FILE = '/data/learn-keras/full_imdb_classifier_tok.pkl'
HYPERPARAMS_FILE = '/data/learn-keras/full_imdb_hyperparams.pkl'


def load_imdb(test=False):
    if test:
        dataroot = '/data/learn-keras/aclImdb/test'
    else:
        dataroot = '/data/learn-keras/aclImdb/train'
    labels = []
    reviews = []
    classifications = {'neg': 0, 'pos': 1}
    for label, target in classifications.items():
        datadir = path.join(dataroot, label)
        for fname in os.listdir(datadir):
            if fname[-4:] == '.txt':
                fpath = path.join(datadir, fname)
                with open(fpath, 'rt') as f:
                    reviews.append(f.read())
                labels.append(target)
    labels = np.array(labels)
    return reviews, labels


def get_train_val(vectors, targets):
    # shuffle the data first
    indices = np.arange(vectors.shape[0])
    np.random.shuffle(indices)
    vectors = vectors[indices]
    targets = targets[indices]

    # Now split it into train/val
    val_size = 10000
    m = hyperparams['train_size']
    return (vectors[:m], targets[:m]), (vectors[m:val_size], targets[m:val_size])


def load_embeddings_matrix(indexes):
    v = hyperparams['vocablen']
    d = hyperparams['dims']

    embeddings_indexes = {}
    with open(f'/data/learn-keras/glove.6B.{d}d.txt', 'rt') as f:
        for line in f:
            flds = line.split()
            word = flds[0]
            embeddings_vec = np.array(flds[1:], dtype=np.float32)
            embeddings_indexes[word] = embeddings_vec

    embeddings_matrix = np.zeros((v, d))
    words_no_embeddings = []
    for word, index in indexes.items():
        if index < v:
            if word in embeddings_indexes:
                embeddings_vec = embeddings_indexes[word]
                embeddings_matrix[index] = embeddings_vec
            else:
                words_no_embeddings.append(word)
    print(f'Unable to find embeddings for {len(words_no_embeddings)} words')
    print(words_no_embeddings[:10])
    return embeddings_matrix


def design_model(embeddings_matrix=None):
    v = hyperparams['vocablen']
    d = hyperparams['dims']
    n = hyperparams['doclen']
    model = Sequential()
    model.add(Embedding(v, d, input_shape=(n,)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    if embeddings_matrix is not None:
        model.layers[0].set_weights([embeddings_matrix])
        model.layers[0].trainable = False
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def train():
    reviews, labels = load_imdb()
    tokenizer = Tokenizer(num_words=hyperparams['vocablen'])
    tokenizer.fit_on_texts(reviews)
    vectors = tokenizer.texts_to_sequences(reviews)
    vectors = pad_sequences(vectors, maxlen=hyperparams['doclen'])
    (x_train, y_train), (x_val, y_val) = get_train_val(vectors, labels)
    if hyperparams['ext_embed']:
        embeddings_matrix = load_embeddings_matrix(tokenizer.word_index)
        model = design_model(embeddings_matrix)
    else:
        model = design_model()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
    model.save(MODEL_FILE)

    with open(TOK_FILE, 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)

    with open(HYPERPARAMS_FILE, 'wb') as f:
        pickle.dump(hyperparams, f, pickle.HIGHEST_PROTOCOL)

    return history


def test():
    with open(TOK_FILE, 'rb') as f:
        tokenizer = pickle.load(f)

    model = load_model(MODEL_FILE)

    with open(HYPERPARAMS_FILE, 'rb') as f:
        hparams = pickle.load(f)
    n = hparams['doclen']

    reviews, y_test = load_imdb(test=True)
    vectors = tokenizer.texts_to_sequences(reviews)
    x_test = pad_sequences(vectors, maxlen=n)
    loss, accuracy = model.evaluate(x_test, y_test)
    print('\n')
    print(f'Test Loss: {loss:.3f}')
    print(f'Test Accuracy: {accuracy:.3f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'test'], help='What action to do - train or test')
    parser.add_argument('-v', '--vocablen', type=int, default=10000, help='Number of top words to include in vocabulary. Defaults to 10000.')
    parser.add_argument('-n', '--doclen', type=int, default=100, help='Maximum number of tokens to consider in each review. Defaults to 100.')
    parser.add_argument('-m', '--trainsize', type=int, default=200, help='Number of reviews to include in training. Defaults to 200.')
    parser.add_argument('-d', '--dims', type=int, default=100, help='Dimensionality of the embeddings. Defaults to 100.')
    parser.add_argument('-e', '--extembed', action='store_true', help='Whether or not to use external embeddings')
    parser.add_argument('-p', '--plot', action='store_true', help='Whether or not to plot.')

    args = parser.parse_args()
    hyperparams['vocablen'] = args.vocablen
    hyperparams['doclen'] = args.doclen
    hyperparams['train_size'] = args.trainsize
    hyperparams['dims'] = args.dims
    hyperparams['ext_embed'] = args.extembed

    print(hyperparams)

    if args.action == 'train':
        history = train()
        if args.plot:
            utils.plot(history)
            plt.show()
    else:
        test()


if __name__ == '__main__':
    main()
