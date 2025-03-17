import os
import os.path as path
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import graph

TMPROOT = '/Users/avilay.parekh/tmp/softmax-iris'


def load_data():
    iris = load_iris()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=0)
    k = len(iris['target_names'])  # number classes
    train_data = (X_train, y_train)
    val_data = (X_val, y_val)
    test_data = (X_test, y_test)
    return train_data, val_data, test_data, k


def create_learner(**kwargs):
    g = kwargs['graph']
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_val = kwargs['X_val']
    y_val = kwargs['y_val']

    def learner(epochs, α_obj):
        val_costs_by_epoch = []
        train_costs_by_epoch = []
        saver = tf.train.Saver()
        checkpoint = path.join(TMPROOT, 'epoch{}_alpha{}.ckpt'.format(epochs, α_obj))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                train = {g.X: X_train, g.y: y_train, g.α: α_obj}
                val = {g.X: X_val, g.y: y_val}
                sess.run(g.optimizer, feed_dict=train)
                J_train = sess.run(g.J, feed_dict=train)
                J_val = sess.run(g.J, feed_dict=val)
                val_costs_by_epoch.append(J_val)
                train_costs_by_epoch.append(J_train)

            val = {g.X: X_val, g.y: y_val}
            J_val = sess.run(g.J, feed_dict=val)
            accuracy_val = sess.run(g.accuracy, feed_dict=val)
            saver.save(sess, checkpoint)
            return J_val, accuracy_val, val_costs_by_epoch, train_costs_by_epoch

    return learner


def plot_learning_curves(title, train_costs, val_costs):
    assert len(train_costs) == len(val_costs), 'Both costs must have same number of elements'
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_costs)), train_costs, label='training costs')
    plt.plot(range(len(val_costs)), val_costs, label='validation costs')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.show()


def save(k, X_test, y_test, scaler):
    testpkl = path.join(TMPROOT, 'iris_test.pkl')
    if path.exists(testpkl):
        os.remove(testpkl)
    pkl = {
        'k': k,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler
    }
    with open(testpkl, 'wb') as f:
        pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)


def main():
    train_data, val_data, test_data, k = load_data()
    scaler = StandardScaler()
    scaler.fit(train_data[0])
    X_train = scaler.transform(train_data[0])
    y_train = train_data[1]
    m, n = X_train.shape

    X_val = scaler.transform(val_data[0])
    y_val = val_data[1]

    X_test = scaler.transform(test_data[0])
    y_test = test_data[1]

    save(k, X_test, y_test, scaler)
    g = graph.build(m, n, k)
    learn = create_learner(graph=g, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    val_cost, val_accuracy, val_costs, train_costs = learn(90, 0.1)
    print('Validation cost: {:.3f}, Validation accuracy: {:.3f}'.format(val_cost, val_accuracy))
    plot_learning_curves('Epoch=700alpha=0.9', train_costs, val_costs)


if __name__ == '__main__':
    main()
