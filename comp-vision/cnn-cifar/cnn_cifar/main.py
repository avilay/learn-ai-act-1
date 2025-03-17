import tensorflow as tf
from .cifardata import *
from .cnn_builder import *


def main():
    X_val, Y_val, filenames = preprocess(*all_valid())
    # print(X_val.shape, Y_val.shape)
    # spot_check(X_val, filenames, 3, samples=[1282, 1283, 1284])

    # Hyperparams
    epochs = 30
    dropout = 0.50

    X = tf.placeholder(tf.float32, [None, 32, 32, 3], 'X')
    Y = tf.placeholder(tf.float32, [None, 10])
    p = tf.placeholder(tf.float32)

    A2 = build_conv(X, filters=32, kernel=(3, 3, 1), pool=(2, 2, 2))
    A3 = build_conv(A2, filters=64, kernel=(3, 3, 1), pool=(2, 2, 2))
    A4 = build_conv(A3, filters=128, kernel=(3, 3, 1), pool=(2, 2, 2))
    A4_flat = flatten(A4)
    A5 = build_conn(A4_flat, 512, p)
    A6 = build_conn(A5, 256, p)
    A7 = build_conn(A6, 128, p)
    Z8 = build_out(A7, 10)

    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z8, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(J)

    correct_pred = tf.equal(tf.argmax(Z8, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            batches_trained = 0
            for X_train, Y_train, _ in train_batches():
                X_train, Y_train, _ = preprocess(X_train, Y_train, None)
                sess.run(optimizer, feed_dict={X: X_train, Y: Y_train, p: 1-dropout})
                batches_trained += 1
                if batches_trained == 10:
                    break

            cost = sess.run(J, feed_dict={X: X_train, Y: Y_train, p: 1.0})
            acc = sess.run(accuracy, feed_dict={X: X_val, Y: Y_val, p: 1.0})
            print('Cost: {0:.3f}\tValidation Accuracy: {1:.3f}'.format(cost, acc))

if __name__ == '__main__':
    main()

