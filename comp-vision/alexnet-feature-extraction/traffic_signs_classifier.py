from collections import namedtuple
import numpy as np
import tensorflow as tf

from alexnet import AlexNet
import traffic_signs
from dataloader import TrafficSignsLoader

HyperParams = namedtuple('HyperParams', ['batch_size', 'epochs', 'learning_rate'])
Model = namedtuple('Model', ['X', 'y', 'probs', 'accuracy', 'optimizer', 'cost'])
CHECKPOINT = '/checkpoints/traffic-signs/alt'


def print_top5(preds):
    ndxs = np.argsort(preds)
    for i in [-1, -2, -3, -4, -5]:
        ndx = ndxs[i]
        name = traffic_signs.class_names[ndx]
        prob_val = preds[ndx]
        print('[{}] {}: {:.3f}'.format(ndx, name, prob_val))


def train(hyper_params, model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X_train, y_train = None, None

        for epoch in range(hyper_params.epochs):
            for X_train, y_train in data.train_batches():
                sess.run(model.optimizer, feed_dict={
                    model.X: X_train, 
                    model.y: y_train})

            # Calculate validation accuracy and cost for this epoch
            tot_val_accuracy = 0
            tot_val_cost = 0
            num_samples = 0
            for X_val, y_val in data.validation_batches():                
                J_out, accuracy_out = sess.run([model.cost, model.accuracy], feed_dict={
                    model.X: X_val,
                    model.y: y_val
                })
                tot_val_accuracy += (accuracy_out * X_val.shape[0])
                tot_val_cost += (J_out * X_val.shape[0])
                num_samples += X_val.shape[0]
            val_accuracy = tot_val_accuracy / num_samples
            val_cost = tot_val_cost / num_samples
            print('Epoch: {} - Validation Accuracy={:.3f} Cost={:.3f}'.format(
                epoch, val_accuracy, val_cost
            ))

        saver.save(sess, CHECKPOINT)


def test(model, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT)
        tot_test_cost = 0
        tot_test_accuracy = 0
        num_samples = 0
        for X_test, y_test in data.test_batches():
            J_out, accuracy_out = sess.run([model.cost, model.accuracy], feed_dict={
                model.X: X_test,
                model.y: y_test
            })
            tot_test_accuracy += (accuracy_out * X_test.shape[0])
            tot_test_cost += (J_out * X_test.shape[0])
            num_samples += X_test.shape[0]

        test_accuracy = tot_test_accuracy / num_samples
        test_cost = tot_test_cost / num_samples
        print('Test Accuracy={:.3f} Cost={:.3f}'.format(epoch, val_accuracy, val_cost))


def main():
    hyper_params = HyperParams(epochs=2, batch_size=128, learning_rate=None)

    X = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, [None])

    X_resized = tf.image.resize_images(X, (227, 227))
    alexnet = AlexNet(X_resized)
    
    # Replace the output layer with a new output layer
    # where the number of classes is 43
    W = tf.Variable(tf.truncated_normal((4096, 43), stddev=0.001))
    b = tf.Variable(tf.zeros(43))
    logits = tf.nn.bias_add(tf.matmul(alexnet.fc7, W), b)
    alexnet.probs = tf.nn.softmax(logits)

    # Stop the gradient from flowing back to layers before fc7
    alexnet.fc7 = tf.stop_gradient(alexnet.fc7)

    # Define the cost function and its optimizer
    Y = tf.one_hot(y, 43)
    J = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer().minimize(J, var_list=[W, b])

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    model = Model(
        X=X, 
        y=y, 
        probs=alexnet.probs, 
        optimizer=optimizer, 
        accuracy=accuracy,
        cost=J)

    data = TrafficSignsLoader()
    data.set_batch_size(hyper_params.batch_size)

    # train(hyper_params, model, data)
    test(model, data)


if __name__ == '__main__':
    main()
