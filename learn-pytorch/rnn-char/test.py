from argparse import ArgumentParser
import pickle
import os.path as path
from collections import namedtuple
import tensorflow as tf
import numpy as np

from graph import RnnGraph
from text_loader import Vocab

HyperParams = namedtuple('HyperParams', [
    'epochs',
    'n_seqs',
    'seq_len',
    'n_layers',
    'lstm_size',
    'dropout',
    'grad_clip',
    'learning_rate'
])


def prepare(args):
    workdir = args.workdir
    if not path.exists(workdir):
        raise RuntimeError('Invalid workdir!')
    ckptroot = path.join(workdir, 'checkpoints')
    vocab_pkl = path.join(workdir, 'vocab.pkl')
    hyperparams_pkl = path.join(workdir, 'hyperparams.pkl')
    costs_pkl = path.join(workdir, 'costs.pkl')

    if (not path.exists(ckptroot) or
            not path.exists(vocab_pkl) or
            not path.exists(hyperparams_pkl) or
            not path.exists(costs_pkl)):
        raise RuntimeError('Some files missing!')

    if args.checkpoint:
        ckpt = path.join(ckptroot, args.checkpoint)
        if not path.exists(ckpt):
            raise RuntimeError('Checkpoint does not exist!')
    else:
        print('*** DEBUG ***', ckptroot)
        ckpt = tf.train.latest_checkpoint(ckptroot)
    return ckpt, vocab_pkl, hyperparams_pkl, costs_pkl


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def main():
    parser = ArgumentParser()
    parser.add_argument('workdir', metavar='WORKDIR', help='Work dir')
    parser.add_argument('-n', '--numchars', type=int, default=100, help='Number of characters to generate')
    parser.add_argument('-f', '--first', default='The', help='First few characters')
    parser.add_argument('-c', '--checkpoint', help='Specific checkpoint file to use')
    args = parser.parse_args()
    ckpt, vocab_pkl, hyperparams_pkl, costs_pkl = prepare(args)

    vocab = Vocab()
    with open(vocab_pkl, 'rb') as vpkl:
        vocab.load(vpkl)

    with open(hyperparams_pkl, 'rb') as hpkl:
        hyper_params = pickle.load(hpkl)
        hyper_params = hyper_params._replace(dropout=0., n_seqs=1, seq_len=1)

    text = args.first
    g = RnnGraph(vocab.size(), hyper_params)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        new_state = sess.run(g.initial_state)
        x = np.zeros((1, 1))
        c = None

        for c in args.first:
            x[0, 0] = vocab.encode(c)[0]
            feed_dict = {
                g.X: x,
                g.initial_state: new_state
            }
            y_hat, new_state = sess.run([g.Y_hat, g.final_state], feed_dict=feed_dict)
            c = pick_top_n(y_hat, vocab.size())

        text += vocab.decode([c])

        for _ in range(args.numchars):
            x[0, 0] = c
            feed_dict = {
                g.X: x,
                g.initial_state: new_state
            }
            y_hat, new_state = sess.run([g.Y_hat, g.final_state], feed_dict=feed_dict)
            c = pick_top_n(y_hat, vocab.size())
            text += vocab.decode([c])

        print('\n******Generated Text')
        print(text)


if __name__ == '__main__':
    main()