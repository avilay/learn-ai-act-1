import os.path as path
import pickle
from graph import *

rootdir = '/Users/avilay.parekh/tmp/rnn-char/dlclass'

# anna = path.join(rootdir, 'anna.tx')
anna = '/Users/avilay.parekh/tmp/trumpeets.txt'
with open(anna, 'r') as f:
    text=f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))

pklfile = path.join(rootdir, 'vocab.pkl')
pkl = {
    'vocab': vocab,
    'vocab_to_int': vocab_to_int,
    'int_to_vocab': int_to_vocab
}
with open(pklfile, 'wb') as f:
    pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)

encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

batches = get_batches(encoded, 10, 50)

batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability

epochs = 20
# Save every N iterations
save_every_n = 200

model = CharRNN(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Use the line below to load a checkpoint and resume training
    # saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        # TODO: Why is the state not carried over from previous epochs?
        new_state = sess.run(model.initial_state)
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {model.inputs: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                feed_dict=feed)

            end = time.time()
            print('Epoch: {}/{}... '.format(e + 1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end - start)))

            if counter % save_every_n == 0:
                ckpt_name = "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size)
                ckpt = path.join(rootdir, 'checkpoints', ckpt_name)
                saver.save(sess, ckpt)

    ckpt_name = "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size)
    ckpt = path.join(rootdir, 'checkpoints', ckpt_name)
    saver.save(sess, ckpt)