import tensorflow as tf
from data_loader import NotMnistLoader


def main():
    data_loader = NotMnistLoader()
    val_files, X_val, Y_val = data_loader.validation_all()
    test_files, X_test, Y_test = data_loader.test_all()

    n = 28*28
    k = 10

    X = tf.placeholder(tf.float32, [None, n])
    Y = tf.placeholder(tf.float32, [None, k])
    W = tf.Variable(tf.truncated_normal([n, k], stddev=0.1))
    b = tf.Variable(tf.zeros(k))
    Z = tf.add(tf.matmul(X, W), b)
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y))
    correct_predictions = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    init = tf.global_variables_initializer()

    # α_train, num_epochs, batch_size
    hyperparams = (
        (0.1, 1, 100),
        # (0.1, 1, 500),
        # (0.1, 1, 1000),
        # (0.1, 1, 5000),
        # (0.1, 1, 10000)
    )

    for α_train, num_epochs, batch_size in hyperparams:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=α_train).minimize(J)
        data_loader.set_batch_size(batch_size)
        print('\nRun with α={}, num_epochs={}, batch_size={}'.format(α_train, num_epochs, batch_size))

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(num_epochs):
                for _, X_train, Y_train in data_loader.train_batches():
                    sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})

                J_val_out = sess.run(J, feed_dict={X: X_val, Y: Y_val})
                accuracy_val_out = sess.run(accuracy, feed_dict={X: X_val, Y: Y_val})
                print('Epoch: {} - Cost: {:.3f} Accuracy: {:.3f}'.format(epoch, J_val_out, accuracy_val_out))

            J_test_out = sess.run(J, feed_dict={X: X_test, Y: Y_test})
            accuracy_test_out = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
            print('Test - Cost: {:.3f} Accuracy: {:.3f}'.format(J_test_out, accuracy_test_out))


if __name__ == '__main__':
    main()
