import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import lenet

DATAROOT = '/Users/avilay.parekh/data/mnist'


def main():
    mnist = input_data.read_data_sets(DATAROOT, reshape=False)

    # Hyperparams
    epochs = 5
    batch_size = 128
    α = 0.001
    μ = 0
    σ = 0.1

    X_train, y_train = mnist.train.images, mnist.train.labels
    X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    X_train, y_train = shuffle(X_train, y_train)

    X = tf.placeholder(tf.float32, [None, 32, 32, 1])
    y = tf.placeholder(tf.int32, [None])
    optimizer, accuracy = lenet.create(X, y, α=α, μ=μ, σ=σ)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for offset in range(0, mnist.train.num_examples, batch_size):
                end = offset + batch_size
                X_train_batch = X_train[offset:end]
                y_train_batch = y_train[offset:end]
                sess.run(optimizer, feed_dict={X: X_train_batch, y: y_train_batch})

            val_acc_out = sess.run(accuracy, feed_dict={X: X_val, y: y_val})
            print('Epoch: {} - Accuracy={:.3f}'.format(epoch, val_acc_out))

        saver.save(sess, './lenet')


if __name__ == '__main__':
    main()
