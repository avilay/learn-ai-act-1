import os.path as path
import tensorflow as tf
import pickle

import graph

TMPROOT = '/Users/avilay.parekh/tmp/softmax-iris'


def main():
    datafile = path.join(TMPROOT, 'iris_test.pkl')
    with open(datafile, 'rb') as f:
        pkl = pickle.load(f)
    k = pkl['k']
    X_test = pkl['X_test']
    m, n = X_test.shape
    y_test = pkl['y_test']
    scaler = pkl['scaler']

    g = graph.build(m, n, k)
    checkpoint = path.join(TMPROOT, 'epoch90_alpha0.1.ckpt')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        # W_out = sess.run(g.W)
        # print(W_out)
        test = {g.X: X_test, g.y: y_test}
        J_test = sess.run(g.J, feed_dict=test)
        accuracy_test = sess.run(g.accuracy, feed_dict=test)
        print('Test cost={:.3f} Test accuracy={:.3f}'.format(J_test, accuracy_test))

if __name__ == '__main__':
    main()
