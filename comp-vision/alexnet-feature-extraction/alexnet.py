from typing import Dict

import numpy as np
import tensorflow as tf

import imagenet


class AlexNet:
    def __init__(self, X):
        self.weights = np.load('/data/traffic-signs/bvlc-alexnet.npy', encoding='latin1').item()  # type: Dict
        self.X = X
        self.conv1 = self._build_conv1()
        self.maxpool1 = self._build_maxpool1()
        self.conv2 = self._build_conv2()
        self.maxpool2 = self._build_maxpool2()
        self.conv3 = self._build_conv3()
        self.conv4 = self._build_conv4()
        self.conv5 = self._build_conv5()
        self.maxpool3 = self._build_maxpool3()
        self.fc6 = self._build_fc6()
        self.fc7 = self._build_fc7()
        self.probs = self._build_fc8()  # output layer

    def _build_conv1(self):
        # kernel: 11 x 11
        # filters: 96
        strides = (1, 4, 4, 1)
        W1 = tf.Variable(self.weights['conv1'][0])  # 11 x 11 x 3 x 96
        b1 = tf.Variable(self.weights['conv1'][1])  # 96
        conv1_Z = tf.nn.bias_add(tf.nn.conv2d(self.X, W1, strides, padding='SAME'), b1)
        conv1 = tf.nn.relu(conv1_Z)  # 57 x 57 x 96
        return conv1

    def _build_maxpool1(self):
        # Normalize
        r = 2
        α = 2e-05
        β = 0.75
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(self.conv1, depth_radius=r, alpha=α, beta=β, bias=bias)

        kernel = (1, 3, 3, 1)
        strides = (1, 2, 2, 1)
        maxpool1 = tf.nn.max_pool(lrn1, ksize=kernel, strides=strides, padding='VALID')  # 28 x 28 x 96
        return maxpool1

    def _build_conv2(self):
        # split convolution
        # kernel: 5 x 5
        # filters: 256
        strides = (1, 1, 1, 1)

        weights = np.split(self.weights['conv2'][0], 2, axis=3)
        b2 = tf.Variable(self.weights['conv2'][1])  # 256
        W2_0 = tf.Variable(weights[0])  # 5 x 5 x 48 x 128
        W2_1 = tf.Variable(weights[1])  # 5 x 5 x 48 x 128

        maxpool1_0, maxpool1_1 = tf.split(self.maxpool1, 2, axis=3)  # 28 x 28 x 48

        conv2_Z0 = tf.nn.conv2d(maxpool1_0, W2_0, strides, padding='SAME')  # 28 x 28 x 128
        conv2_Z1 = tf.nn.conv2d(maxpool1_1, W2_1, strides, padding='SAME')  # 28 x 28 x 128

        conv2_Z = tf.nn.bias_add(tf.concat([conv2_Z0, conv2_Z1], axis=3), b2)
        conv2 = tf.nn.relu(conv2_Z)  # 28 x 28 x 256
        return conv2

    def _build_maxpool2(self):
        r = 2
        α = 2e-05
        β = 0.75
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(self.conv2, depth_radius=r, alpha=α, beta=β, bias=bias)

        kernel = (1, 3, 3, 1)
        strides = (1, 2, 2, 1)
        maxpool2 = tf.nn.max_pool(lrn2, ksize=kernel, strides=strides, padding='VALID')  # 13 x 13 x 256
        return maxpool2

    def _build_conv3(self):
        # kernel: 3 x 3
        # filters: 384
        strides = (1, 1, 1, 1)
        W3 = tf.Variable(self.weights['conv3'][0])  # 3 x 3 x 256 x 384
        b3 = tf.Variable(self.weights['conv3'][1])  # 384
        conv3_Z = tf.nn.bias_add(tf.nn.conv2d(self.maxpool2, W3, strides=strides, padding='SAME'), b3)
        conv3 = tf.nn.relu(conv3_Z)  # 13 x 13 x 384
        return conv3

    def _build_conv4(self):
        # split convolution
        # kernel: 3 x 3
        # filters: 384
        strides = (1, 1, 1, 1)

        weights = np.split(self.weights['conv4'][0], 2, axis=3)
        W4_0 = tf.Variable(weights[0])  # 3 x 3 x 192 x 192
        W4_1 = tf.Variable(weights[1])  # 3 x 3 x 192 x 192
        b4 = tf.Variable(self.weights['conv4'][1])  # 384

        conv3_0, conv3_1 = tf.split(self.conv3, 2, axis=3)  # 13 x 13 x 192

        conv4_Z0 = tf.nn.conv2d(conv3_0, W4_0, strides=strides, padding='SAME')  # 13 x 13 x 192
        conv4_Z1 = tf.nn.conv2d(conv3_1, W4_1, strides=strides, padding='SAME')  # 13 x 13 x 192

        conv4_Z = tf.nn.bias_add(tf.concat([conv4_Z0, conv4_Z1], axis=3), b4)
        conv4 = tf.nn.relu(conv4_Z)  # 13 x 13 384
        return conv4

    def _build_conv5(self):
        # split convolution
        # kernel: 3 x 3
        # filters: 256
        strides = (1, 1, 1, 1)

        weights = np.split(self.weights['conv5'][0], 2, axis=3)
        W5_0 = tf.Variable(weights[0])  # 3 x 3 x 192 x 128
        W5_1 = tf.Variable(weights[1])  # 3 x 3 x 192 x 128
        b5 = tf.Variable(self.weights['conv5'][1])  # 256

        conv4_0, conv4_1 = tf.split(self.conv4, 2, axis=3)  # 13 x 13 x 192

        conv5_Z0 = tf.nn.conv2d(conv4_0, W5_0, strides=strides, padding='SAME')  # 13 x 13 x 128
        conv5_Z1 = tf.nn.conv2d(conv4_1, W5_1, strides=strides, padding='SAME')  # 13 x 13 x 128

        conv5_Z = tf.nn.bias_add(tf.concat([conv5_Z0, conv5_Z1], axis=3), b5)
        conv5 = tf.nn.relu(conv5_Z)  # 13 x 13 x 256
        return conv5

    def _build_maxpool3(self):
        kernel = (1, 3, 3, 1)
        strides = (1, 2, 2, 1)
        maxpool3 = tf.nn.max_pool(self.conv5, ksize=kernel, strides=strides, padding='VALID')  # 6 x 6 x 256
        return maxpool3

    def _build_fc6(self):
        # num_nodes = 4096
        maxpool3_flat = tf.reshape(self.maxpool3, [-1, 6 * 6 * 256])
        W6 = tf.Variable(self.weights['fc6'][0])  # 9216 x 4096
        b6 = tf.Variable(self.weights['fc6'][1])  # 4096
        Z6 = tf.nn.bias_add(tf.matmul(maxpool3_flat, W6), b6)
        fc6 = tf.nn.relu(Z6)  # 4096
        return fc6

    def _build_fc7(self):
        # num_nodes = 4096
        W7 = tf.Variable(self.weights['fc7'][0])  # 4096 x 4096
        b7 = tf.Variable(self.weights['fc7'][1])  # 4096
        Z7 = tf.nn.bias_add(tf.matmul(self.fc6, W7), b7)
        fc7 = tf.nn.relu(Z7)  # 4096
        return fc7

    def _build_fc8(self):
        # num_nodes = 1000
        W8 = tf.Variable(self.weights['fc8'][0])  # 4096 x 1000
        b8 = tf.Variable(self.weights['fc8'][1])  # 1000
        logits = tf.nn.bias_add(tf.matmul(self.fc7, W8), b8)
        fc8 = tf.nn.softmax(logits)  # 1000
        return fc8


def main():
    from scipy.misc import imread

    # Get rid of the alpha channel when reading the image in
    poodle = imread('res/poodle.png')[:, :, :3]
    weasel = imread('res/weasel.png')[:, :, :3]

    # Run Inference
    poodle_norm = poodle - np.mean(poodle)
    weasel_norm = weasel - np.mean(weasel)

    X = tf.placeholder(tf.float32, (None, 227, 227, 3))
    alexnet = AlexNet(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        poodle_pred, weasel_pred = sess.run(alexnet.probs, feed_dict={X: [poodle_norm, weasel_norm]})

    print('\nTop 5 predictions for the poolde image')
    poodle_ndxs = np.argsort(poodle_pred)
    for i in [-1, -2, -3, -4, -5]:
        ndx = poodle_ndxs[i]
        name = imagenet.class_names[ndx]
        prob_val = poodle_pred[ndx]
        print('[{}] {}: {:.3f}'.format(ndx, name, prob_val))

    print('\nTop 5 predictions for the weasel image')
    weasel_ndxs = np.argsort(weasel_pred)
    for i in [-1, -2, -3, -4, -5]:
        ndx = weasel_ndxs[i]
        name = imagenet.class_names[ndx]
        prob_val = weasel_pred[ndx]
        print('[{}] {}: {:.3f}'.format(ndx, name, prob_val))


if __name__ == '__main__':
    main()
