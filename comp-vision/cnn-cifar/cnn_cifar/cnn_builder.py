import tensorflow as tf
import numpy as np


def build_conv(A0, filters, kernel, pool=None):
    """
    Creates a tensor for a conv2d layer with max pool
    :param A0: previous layer's output
    :param filters: Number of filters
    :param kernel: (kernel_width, kernel_height, stride)
    :param pool: (pool_width, pool_height, stride)
    :return: tensor
    """
    input_depth = A0.get_shape().as_list()[-1]
    # W0 = tf.Variable(tf.random_normal([kernel[0], kernel[1], input_depth, filters]))
    # b0 = tf.Variable(tf.random_normal([filters]))
    W0 = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], input_depth, filters], stddev=0.1))
    b0 = tf.Variable(tf.constant(0.1, shape=[filters]))
    strides = [1, kernel[2], kernel[2], 1]
    Z1 = tf.add(tf.nn.conv2d(A0, W0, strides=strides, padding='SAME'), b0)
    strides = [1, pool[2], pool[2], 1]
    A1 = tf.nn.max_pool(tf.nn.relu(Z1), ksize=[1, pool[0], pool[1], 1], strides=strides, padding='SAME')
    return A1


def build_conn(A0, units, keep_prob):
    """
    Creates a tensor for a fully connected layer with dropout
    :param A0: previous layer's output
    :param units: number of units in this layer
    :param dropout: whether or not to add dropout
    :return: tensor
    """
    input_units = A0.get_shape().as_list()[-1]
    # W0 = tf.Variable(tf.random_normal([input_units, units]))
    # b0 = tf.Variable(tf.random_normal([units]))
    W0 = tf.Variable(tf.truncated_normal([input_units, units], stddev=0.1))
    b0 = tf.Variable(tf.constant(0.1, shape=[units]))
    Z1 = tf.add(tf.matmul(A0, W0), b0)
    A1 = tf.nn.relu(Z1)
    A1_dropped = tf.nn.dropout(A1, keep_prob)
    return A1_dropped


def build_out(A0, k):
    input_units = A0.get_shape().as_list()[-1]
    # W0 = tf.Variable(tf.random_normal([input_units, k]))
    # b0 = tf.Variable(tf.random_normal([k]))
    W0 = tf.Variable(tf.truncated_normal([input_units, k], stddev=0.1))
    b0 = tf.Variable(tf.constant(0.1, shape=[k]))
    Z1 = tf.add(tf.matmul(A0, W0), b0)
    return Z1


def flatten(A):
    A_shape = A.get_shape().as_list()
    flat_size = np.prod(A_shape[1:])
    A_flat = tf.reshape(A, [-1, flat_size])
    return A_flat

