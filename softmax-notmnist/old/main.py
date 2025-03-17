import tensorflow as tf
import os.path as path
import pickle
import matplotlib.pyplot as plt
from data_loader import NotMnistLoader
from data_loader_dbg import NotMnistLoaderDgb


def main():
    # data_loader = NotMnistLoader(flatten=True)
    data_loader = NotMnistLoader()
    # data_loader = NotMnistLoaderDgb()

    print('Loading validation data')
    val_files, X_val, Y_val = data_loader.validation_all()
    # data_loader.show(5, val_files, X_val, Y_val)
    # plt.show()

    n = 28*28  # number of features
    k = 10     # number of classes

    X = tf.placeholder(tf.float32, [None, n])
    Y = tf.placeholder(tf.float32, [None, k])
    W = tf.Variable(tf.truncated_normal([n, k], stddev=0.1))
    b = tf.Variable(tf.zeros([k]))
    # W = tf.Variable(tf.random_normal([n, k]))
    # b = tf.Variable(tf.random_normal([k]))
    logits = tf.add(tf.matmul(X, W), b)
    # H = tf.nn.softmax(logits)

    # cross_entropy = - tf.multiply(Y, tf.log(H))
    # J = tf.reduce_mean(cross_entropy) / m
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    α = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=α).minimize(J)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    # Hyperparams
    num_epochs = 1
    α_train = 0.1

    # print('Training model')
    # # Train the model
    # with tf.Session() as sess:
    #     sess.run(init)
    #
    #     for epoch in range(num_epochs):
    #         batch_num = 0
    #         for _, X_train, Y_train in data_loader.train_batches():
    #             feed_dict = {
    #                 X: X_train,
    #                 Y: Y_train,
    #                 α: α_train
    #             }
    #             sess.run(optimizer, feed_dict=feed_dict)
    #
    #             # Print batch stats
    #             if batch_num % 50 == 0:
    #                 train_J_out = sess.run(J, feed_dict={X: X_train, Y: Y_train})
    #                 val_accuracy_out = sess.run(accuracy, feed_dict={X: X_val, Y: Y_val})
    #                 print('Epoch: {} Batch: {} => Training Cost: {:.3f} Validation Accuracy: {:.3f}'.format(
    #                     epoch, batch_num, train_J_out, val_accuracy_out))
    #
    #             batch_num += 1

    print('Testing model')
    # Measure test accuracy
    test_files, X_test, Y_test = data_loader.test_all()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            for _, X_train, Y_train in data_loader.train_batches():
                feed_dict = {
                    X: X_train,
                    Y: Y_train,
                    α: α_train
                }
                sess.run(optimizer, feed_dict=feed_dict)

        test_accuracy_out = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
        print('Test Accuracy: {:.3f}'.format(test_accuracy_out))


if __name__ == '__main__':
    main()
