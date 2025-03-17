import tensorflow as tf
from tensorflow.contrib.layers import flatten


def create(X, y, α=0, μ=0, σ=0):
    Y = tf.one_hot(y, 10)

    # Layer 2 is a convolutional layer with output of 28x28x6
    # followed by a pooling layer with output of 14x14x6
    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean=μ, stddev=σ), name='W1')
    b1 = tf.Variable(tf.zeros([6]), name='b1')
    Z2 = tf.nn.bias_add(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID'), b1)
    A2_full = tf.nn.relu(Z2)
    A2 = tf.nn.max_pool(A2_full, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 3 is a convolutional layer with output of 10x10x16
    # followed by a pooling layer with output of 5x5x16
    W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=μ, stddev=σ), name='W2')
    b2 = tf.Variable(tf.zeros([16]), name='b2')
    Z3 = tf.nn.bias_add(tf.nn.conv2d(A2, W2, strides=[1, 1, 1, 1], padding='VALID'), b2)
    A3_full = tf.nn.relu(Z3)
    A3 = tf.nn.max_pool(A3_full, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Layer 4 is a fully connected layer with 120 units (input is 5x5x16 = 400 units)
    A3_flat = flatten(A3)
    W3 = tf.Variable(tf.truncated_normal([400, 120], mean=μ, stddev=σ), name='W3')
    b3 = tf.Variable(tf.zeros([120]), name='b3')
    Z4 = tf.nn.bias_add(tf.matmul(A3_flat, W3), b3)
    A4 = tf.nn.relu(Z4)

    # Layer 5 is a fully connected layer with 84 units
    W4 = tf.Variable(tf.truncated_normal([120, 84], mean=μ, stddev=σ), name='W4')
    b4 = tf.Variable(tf.zeros([84]), name='b4')
    Z5 = tf.nn.bias_add(tf.matmul(A4, W4), b4)
    A5 = tf.nn.relu(Z5)

    # Layer 6 is a fully connected output layer
    W5 = tf.Variable(tf.truncated_normal([84, 10], mean=μ, stddev=σ), name='W5')
    b5 = tf.Variable(tf.zeros([10]), name='b5')
    Z6 = tf.nn.bias_add(tf.matmul(A5, W5), b5)
    print(Z6.get_shape())
    print(Y.get_shape())

    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z6, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=α).minimize(J)

    correct_pred = tf.equal(tf.argmax(Z6, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return optimizer, accuracy
