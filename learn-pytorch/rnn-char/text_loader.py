"""
All data methods return a 2-element tuple X, Y.
X is the input matrix. This is different from the traditional X matrix.
Technically each element of X is a single instance in itself because the
number of "features" is one. However, we feed seq_len number of chars to the
RNN at the same time. X is then composed of seq_len columns. Because each element
of X has a corresponding output, Y is also composed of seq_len columns.

Each element of X or Y is not the actual character, but rather the character ID.

The batch methods return batch_size number of rows at a time.

Lets say the text composed of -
    abcdefghijklmnopqr

For seq_len = 3 and batch_size = 2, the text would be split as follows -

Batch 1
-------
X = [[abc],
     [def]]
Y = [[bcd],
     [efg]]

Batch 2
-------
X = [[ghi],
     [jkl]]
Y = [[hij],
     [klm]]

Batch 3
-------
X = [[mno]]
Y = [[nop]]

It is important to feed the RNN characters in the same order as they appear in the text.
"""
from typing import Dict
from typing import List

import numpy as np
import pickle


class Vocab:
    def __init__(self):
        self._contents = None
        self._id_to_char = None  # type: List[str]
        self._char_to_id = None  # type: Dict[str, int]

    def size(self):
        return len(self._id_to_char)

    def read(self, filelike):
        if self._contents or self._id_to_char or self._char_to_id:
            raise RuntimeError('Vocab has already been initialized!')
        self._contents = filelike.read()
        self._id_to_char = list(set(self._contents))  # chars[id] = char
        self._char_to_id = {char: idx for idx, char in enumerate(self._id_to_char)}

    def load(self, filelike):
        if self._contents or self._id_to_char or self._char_to_id:
            raise RuntimeError('Vocab has already been initialized!')
        pkl = pickle.load(filelike)
        self._contents = pkl['contents']
        self._id_to_char = pkl['id_to_char']
        self._char_to_id = pkl['char_to_id']

    def save(self, filelike):
        pkl = {
            'contents': self._contents,
            'id_to_char': self._id_to_char,
            'char_to_id': self._char_to_id
        }
        pickle.dump(pkl, filelike, pickle.HIGHEST_PROTOCOL)

    def encode(self, text):
        return np.array([self._char_to_id[char] for char in text])

    def decode(self, ids):
        return ''.join([self._id_to_char[id_] for id_ in ids])

    def encode_all(self):
        return self.encode(self._contents)


class TextLoader:
    def __init__(self, vocab, seq_len, n_seqs, val_frac=0.1):
        self._n_seqs = n_seqs
        self._seq_len = seq_len
        self._vocab = vocab

        x = self._vocab.encode_all()
        y = x[1:]

        # x and y must be exactly splittable in chunks that are seq_len in size.
        # Also, x and y must be the same length. Right now y is 1 less than x.
        # This means len(x) and len(y) must be equal and be exactly divisible by seq_len.
        # To make this happen we'll need to throw away some chars at the end.

        len_x = len(x)
        len_y = len(y)
        new_len_x = self._seq_len * (len_x // self._seq_len)
        new_len_y = self._seq_len * (len_y // self._seq_len)
        new_len = min(new_len_x, new_len_y)
        x = x[:new_len]
        y = y[:new_len]
        assert len(x) == len(y), 'x and y are not the same length!'
        assert len(x) % self._seq_len == 0, 'x is not splittable!'
        assert len(y) % self._seq_len == 0, 'y is not splittable!'

        num_splits = new_len // self._seq_len
        X = np.array(np.split(x, num_splits))
        Y = np.array(np.split(y, num_splits))

        val_size = self._n_seqs * (int(X.shape[0] * val_frac) // self._n_seqs)
        train_size = self._n_seqs * (int(X.shape[0] * (1 - val_frac)) // self._n_seqs)
        self.X_train = X[:train_size, :]
        self.Y_train = Y[:train_size, :]
        self.X_val = X[train_size:train_size + val_size, :]
        self.Y_val = Y[train_size:train_size + val_size, :]

    def vocab_size(self):
        return self._vocab.size()

    def validation_all(self):
        return self.X_val, self.Y_val

    def validation_batches(self):
        m = self.X_val.shape[0]
        for i in range(0, m, self._n_seqs):
            yield self.X_val[i:i+self._n_seqs], self.Y_val[i:i + self._n_seqs]

    def train_all(self):
        return self.X_train, self.Y_train

    def train_batches(self):
        m = self.X_train.shape[0]
        for i in range(0, m, self._n_seqs):
            yield self.X_train[i:i+self._n_seqs], self.Y_train[i:i + self._n_seqs]

    def train_batches_dbg(self):
        n_seqs = self._n_seqs
        n_steps = self._seq_len
        arr = self._vocab.encode_all()

        batch_size = n_seqs * n_steps
        n_batches = len(arr) // batch_size

        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size]

        # Reshape into n_seqs rows
        arr = arr.reshape((n_seqs, -1))

        for n in range(0, arr.shape[1], n_steps):
            # The features
            x = arr[:, n:n + n_steps]
            # The targets, shifted by one
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y


def main1():
    with open('/Users/avilay.parekh/data/anna.txt', 'rt') as f:
        dl = TextLoader(f, 100, 10)

    for i, (X_batch, Y_batch) in enumerate(dl.train_batches(), start=1):
        print('Batch: {}, {}, {}'.format(i, X_batch.shape, Y_batch.shape))

    for i, (X_val, Y_val) in enumerate(dl.validation_batches(), start=1):
        print('Batch: {}, {}, {}'.format(i, X_val.shape, Y_val.shape))


def main():
    import io
    contents = 'abcdefghijklmnopqr'
    f = io.StringIO(contents)
    vocab = Vocab()
    vocab.read(f)
    dl = TextLoader(vocab, 3, n_seqs=2, val_frac=0.0)
    print('Vocab: {}'.format(vocab.encode_all()))
    print('My batches')
    for X, Y in dl.train_batches():
        print('X --')
        print(X)
        print('Y --')
        print(Y)
        print('------------\n')

    print('Dbg batches')
    for X, Y in dl.train_batches_dbg():
        print('X --')
        print(X)
        print('Y --')
        print(Y)
        print('------------\n')


if __name__ == '__main__':
    main()
