import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet

DATAROOT = '/Users/avilay.parekh/data/mnist'
BATCH_SIZE = 128


def main():
    mnist = input_data.read_data_sets(DATAROOT, reshape=False)
    X_test, y_test = mnist.test.images, mnist.test.labels
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    X = tf.placeholder(tf.float32, [None, 32, 32, 1])
    y = tf.placeholder(tf.int32, [None])
    optimizer, accuracy = lenet.create(X, y)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        tot_accuracy = 0
        for offset in range(0, mnist.test.num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            X_batch, y_batch = X_test[offset:end], y_test[offset:end]
            accuracy_out = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
            tot_accuracy += (accuracy_out * len(X_batch))

        print('Test Accuracy: {:.3f}'.format(tot_accuracy/mnist.test.num_examples))


if __name__ == '__main__':
    main()
