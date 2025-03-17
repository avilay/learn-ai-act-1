import os.path as path
import pickle
from graph import *


rootdir = '/Users/avilay.parekh/tmp/tmp200'

pklfile = path.join(rootdir, 'vocab.pkl')
with open(pklfile, 'rb') as f:
    pkl = pickle.load(f)
vocab = pkl['vocab']
vocab_to_int = pkl['vocab_to_int']
int_to_vocab = pkl['int_to_vocab']


def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    print(checkpoint)
    samples = [c for c in prime]
    encoded_samples = [vocab_to_int[c] for c in prime]
    model = CharRNN(len(vocab), lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            encoded_samples.append(c)
            samples.append(int_to_vocab[c])

    print(encoded_samples)
    return ''.join(samples)


def main():
    # ckroot = path.join(rootdir, 'checkpoints')
    # print(ckroot)
    lstm_size = 512
    # checkpoint = tf.train.latest_checkpoint(ckroot)
    checkpoint = path.join(rootdir, 'checkpoints', 'i39600_l512.ckpt')
    print(checkpoint)
    samp = sample(checkpoint, 20, lstm_size, len(vocab), prime="Far")
    print(samp)


if __name__ == '__main__':
    main()


