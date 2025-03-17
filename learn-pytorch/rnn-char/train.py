from argparse import ArgumentParser
import os
import shutil
import os.path as path
from datetime import datetime
import time
from collections import namedtuple
import pickle
from configparser import ConfigParser
import tensorflow as tf
import avilabsutils as utils

from text_loader import Vocab, TextLoader
from graph import RnnGraph

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

DEFAULT_HPARAMS = HyperParams(
    epochs=2,
    n_seqs=100,
    seq_len=100,
    n_layers=2,
    lstm_size=512,
    dropout=0.5,
    grad_clip=5,
    learning_rate=0.001
)


def prepare(workroot):
    if not path.exists(workroot):
        raise RuntimeError('Invalid workroot!')

    ts = str(int(datetime.now().timestamp()))[-6:]
    workdir = path.join(workroot, ts)
    if path.exists(workdir):
        shutil.rmtree(workdir)

    ckptroot = path.join(workdir, 'checkpoints')
    os.makedirs(ckptroot)

    vocab_pkl = path.join(workdir, 'vocab.pkl')
    hyperparams_pkl = path.join(workdir, 'hyperparams.pkl')
    costs_pkl = path.join(workdir, 'costs.pkl')

    return ckptroot, vocab_pkl, hyperparams_pkl, costs_pkl


def gen_hyper_params(hfilename=None):
    if not hfilename:
        yield DEFAULT_HPARAMS
    else:
        if not path.exists(hfilename):
            raise RuntimeError('Invalid hyper params file!')
        hfile = ConfigParser()
        hfile.read(hfilename)
        epochs_vals = [int(epochs) for epochs in hfile['DEFAULT']['epochs'].split(',')]
        n_seqs_vals = [int(n_seqs) for n_seqs in hfile['DEFAULT']['n_seqs'].split(',')]
        seq_len_vals = [int(seq_len) for seq_len in hfile['DEFAULT']['seq_len'].split(',')]
        n_layers_vals = [int(n_layers) for n_layers in hfile['DEFAULT']['n_layers'].split(',')]
        lstm_size_vals = [int(lstm_size) for lstm_size in hfile['DEFAULT']['lstm_size'].split(',')]
        dropout_vals = [float(dropout) for dropout in hfile['DEFAULT']['dropout'].split(',')]
        grad_clip_vals = [int(grad_clip) for grad_clip in hfile['DEFAULT']['grad_clip'].split(',')]
        learning_rate_vals = [float(learning_rate) for learning_rate in hfile['DEFAULT']['learning_rate'].split(',')]

        for epochs in epochs_vals:
            for n_seqs in n_seqs_vals:
                for seq_len in seq_len_vals:
                    for n_layers in n_layers_vals:
                        for lstm_size in lstm_size_vals:
                            for dropout in dropout_vals:
                                for grad_clip in grad_clip_vals:
                                    for learning_rate in learning_rate_vals:
                                        yield HyperParams(
                                            epochs=epochs,
                                            n_seqs=n_seqs,
                                            seq_len=seq_len,
                                            n_layers=n_layers,
                                            lstm_size=lstm_size,
                                            dropout=dropout,
                                            grad_clip=grad_clip,
                                            learning_rate=learning_rate
                                        )


def main():
    parser = ArgumentParser()
    parser.add_argument('textfile', metavar='TEXTFILE', help='Input text file')
    parser.add_argument('workroot', metavar='WORKROOT', help='Where state files will be written')
    parser.add_argument('-y', '--hyperparams', help='ini file with hyperparams')
    args = parser.parse_args()

    vocab = Vocab()
    if not path.exists(args.textfile):
        raise RuntimeError('Invalid file path!')
    with open(args.textfile, 'rt') as f:
        vocab.read(f)

    for hyper_params in gen_hyper_params(args.hyperparams):
        print('Starting with {}'.format(hyper_params))
        ckptroot, vocab_pkl, hyperparams_pkl, costs_pkl = prepare(args.workroot)
        text_loader = TextLoader(vocab, hyper_params.seq_len, hyper_params.n_seqs, val_frac=0.0)
        g = RnnGraph(vocab.size(), hyper_params)

        training_costs = []
        saver = tf.train.Saver(max_to_keep=100)
        sess_start = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_state = sess.run(g.initial_state)
            for epoch in range(1, hyper_params.epochs + 1):
                for batch, (X_in, Y_in) in enumerate(text_loader.train_batches(), start=1):
                    print(f'Epoch: {epoch} Batch: {batch}')
                    feed_dict = {
                        g.X: X_in,
                        g.Y: Y_in,
                        g.initial_state: new_state
                    }
                    J_out, new_state, _ = sess.run([g.J, g.final_state, g.optimizer], feed_dict=feed_dict)
                    training_costs.append(J_out)
                    if batch % 200 == 0:
                        utils.print_code('Epoch: {}\tBatch:{}\tCost:{:.3f}'.format(epoch, batch, J_out))
                        ckptname = 'e{}_b{}.ckpt'.format(epoch, batch)
                        ckpt = path.join(ckptroot, ckptname)
                        saver.save(sess, ckpt)

            utils.print_code('Epoch: {}\tBatch:{}\tCost:{:.3f}'.format(epoch, batch, J_out))
            ckptname = 'e{}_b{}.ckpt'.format(epoch, batch)
            ckpt = path.join(ckptroot, ckptname)
            saver.save(sess, ckpt)

        sess_end = time.time()

        utils.print_success('Finished hyper params session in {} minutes'.format((sess_end - sess_start)//60))
        with open(vocab_pkl, 'wb') as vocabpkl:
            vocab.save(vocabpkl)
        with open(hyperparams_pkl, 'wb') as hpkl:
            pickle.dump(hyper_params, hpkl, pickle.HIGHEST_PROTOCOL)
        with open(costs_pkl, 'wb') as cpkl:
            pickle.dump(training_costs, cpkl, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
