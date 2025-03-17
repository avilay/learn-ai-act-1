import os.path as path
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATAROOT = '/Users/avilay.parekh/data/mnist'

"""
Consider a 28x28 grayscale image which has a length and breadth of 28 each but a depth of only 1. Think of a *filter*
of size 5x5 as a square pyramid with its base on the input image and its apex on the output unit. Lets say this is a
_charmed_ filter that learns stright lines in the image. Now slide this pyramid over the entire input image moving it
1 pixel/unit at a time. The number of output units are 26x26 (28 - 3 + 1). But each output unit has the same weights.
The base of the pyramid actually represents the weights connecting the 5x5 input units to a single output unit, i.e,
the pyramid base represents 25 weights. And as usual each output unit will have a bias, but instead of having 26x26
biases, all output units share the same bias, i.e, the apex of the pyramid represents the bias. So, even though there
are 26x26 output units partially connected to 28x28 input units, we only need 5x5 + 1 parameters. This aspect of
all output units sharing the same weights is called *kernel sharing*.

Now consider a 28x28 color image with the usual 3 RGB channels. The depth of the input image is now 3. Our charmed
pyramid is still learning straight lines, but it is learning them in all three input layers. The base of the pyramid
is now thicker. Instead of 5x5 pixels, it connects 5x5x3 pixels/units to a single output unit. It is important to note
that the 3 input patches do not share weights, each input patch has its own weight, which is why we end up with
5x5x3 weights. Now, as usual, we slide the (now thick based) pyramid across the input image and construct our output
layer. All the output units still share the same 5x5x3 weights and have a single bias. So now we have an input layer
of 28x28x3 units, partially connected to 26x26 output units, but we only need 5x5x3 + 1 parameters.

Just learning straight lines in an image is not enough. We need to add more filters - lets say we add 5 more filters
_strange_, _top_, _bottom_, _up_, and _down_ that learn different aspects of the image. Think of each of these filters
as a pyramid each that is being slid over the entire input image. Because we now have 6 filters we need 6 x (5x5x3 + 1)
parameters. And we'll have six different 26x26 output sheets. We'll just stack these sheets one after the other.

 |||
 |||    ||||||
 ||| \  ||||||
 ||| /  ||||||
 |||
"""


def main():
    mnist = input_data.read_data_sets(DATAROOT, one_hot=True, reshape=False)

    # Hyperparams
    α = 0.00001
    epochs = 10
    train_batch_size = 128
    validation_batch_size = 256

    # Network Parameters
    k = 10  # MNIST total classes (0-9 digits)
    keep_prob = 0.75  # Dropout, probability to keep units

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, k])
    p = tf.placeholder(tf.float32)

    # L2 is a convolution + max-pool layer -
    # - same padding
    # - 32 filters
    # - each filter having a 5x5x1 kernel
    # - stride length of 1
    # - RELU as the activation function
    # - pool size of 2x2
    # 28x28x1 image --conv--> 28x28x32 --mp--> 14x14x32
    # Shape of weight is patch_height, patch_width, input_depth, output_depth
    W1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
    # Shape of bias is output depth
    b1 = tf.Variable(tf.random_normal([32]))
    Z2 = tf.nn.bias_add(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'), b1)
    A2 = tf.nn.max_pool(tf.nn.relu(Z2), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # L3 is a convolution layer
    # - same padding
    # - 64 filters
    # - each filter having a 5x5x32 kernel
    # - stride length of 1
    # - RELU as the activation function
    # - pool size of 2x2
    # 14x14x32 --conv--> 14x14x64 --mp--> 7x7x64
    W2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
    b2 = tf.Variable(tf.random_normal([64]))
    Z3 = tf.nn.bias_add(tf.nn.conv2d(A2, W2, strides=[1, 1, 1, 1], padding='SAME'), b2)
    A3 = tf.nn.max_pool(tf.nn.relu(Z3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # L4 is a fully connected layer with -
    # - 1024 units
    # - RELU as the activation function
    # - Dropout as the regularization mechanism
    W3 = tf.Variable(tf.random_normal([7*7*64, 1024]))
    b3 = tf.Variable(tf.random_normal([1024]))
    A3_flat = tf.reshape(A3, [-1, W3.get_shape().as_list()[0]])
    Z4 = tf.add(tf.matmul(A3_flat, W3), b3)
    A4 = tf.nn.relu(Z4)
    A4_dropped = tf.nn.dropout(A4, p)

    # L5 is the output layer, another fully connected layer with -
    # - softmax as the activation
    W4 = tf.Variable(tf.random_normal([1024, k]))
    b4 = tf.Variable(tf.random_normal([k]))
    Z5 = tf.add(tf.matmul(A4_dropped, W4), b4)
    # H = tf.nn.softmax(Z5)

    # J = - tf.reduce_sum(tf.multiply(Y, tf.log(H))) / m
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z5, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=α).minimize(J)

    correct_pred = tf.equal(tf.argmax(Z5, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    num_batches = int(mnist.train.num_examples//train_batch_size)
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            for batch in range(num_batches):
                X_batch, Y_batch = mnist.train.next_batch(train_batch_size)
                sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch, p: keep_prob})

            # Calculate epoch loss and accuracy
            loss = sess.run(J, feed_dict={X: X_batch, Y: Y_batch, p: 1.})
            X_val = mnist.validation.images[:validation_batch_size]
            Y_val = mnist.validation.labels[:validation_batch_size]
            valid_acc = sess.run(accuracy, feed_dict={X: X_val, Y: Y_val, p: 1.})

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                    epoch + 1,
                    batch + 1,
                    loss,
                    valid_acc))

        # Calculate Test Accuracy
        X_test = mnist.test.images[validation_batch_size]
        Y_test = mnist.test.labels[validation_batch_size]
        test_acc = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test, p: 1.})
        print('Testing Accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    main()
