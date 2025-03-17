from collections import namedtuple
import tensorflow as tf


Graph = namedtuple('Graph', ['X', 'y', 'α', 'J', 'optimizer', 'accuracy', 'W'])


def build(m, n, k):
    X = tf.placeholder(tf.float32, [None, n])
    y = tf.placeholder(tf.int32, [None])
    α = tf.placeholder(tf.float32)
    Y = tf.one_hot(y, k)
    with tf.variable_scope('Demo'):
        W = tf.Variable(tf.truncated_normal([n, k], stddev=0.1), name='Li')
        b = tf.Variable(tf.zeros(k), name='Avi')
    Z = tf.add(tf.matmul(X, W), b)
    # Y_hat = tf.nn.softmax(Z)
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=α).minimize(J)
    correct_predictions = tf.equal(tf.argmax(Z, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return Graph(X=X, y=y, α=α, J=J, optimizer=optimizer, accuracy=accuracy, W=W)
