import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile
import math
import matplotlib.image as mpimg
import os.path as path


def normalize_grayscale(img):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + (((img - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


def load_imgs(filepath):
    # Comment here on what this does
    features = []
    labels = []
    filenames_pbar = tqdm(os.listdir(filepath), unit='files')
    for filename in filenames_pbar:
        if filename.endswith('.png'):
            imgpath = path.join(filepath, filename)
            # img = mpimg.imread(imgpath)
            img = Image.open(imgpath)
            img.load()
            feature = np.array(img, dtype=np.float32).flatten()
            label = path.basename(imgpath)[0]
            features.append(feature)
            labels.append(label)

    return np.array(features), np.array(labels)


# Hyperparams
epochs = 1
batch_size = 100
learning_rate = 0.1

# Get the features and labels from the zip files
all_X_train_val_raw, all_y_train_val_raw = load_imgs('/Users/avilay.parekh/data/notMNIST/notMNIST_train')
X_test_raw, y_test_raw = load_imgs('/Users/avilay.parekh/data/notMNIST/notMNIST_test')

# Limit the amount of data to work with a docker container
docker_size_limit = 150000
X_train_val_raw, y_train_val_raw = resample(all_X_train_val_raw, all_y_train_val_raw, n_samples=docker_size_limit)


X_train_val = normalize_grayscale(X_train_val_raw)
X_test = normalize_grayscale(X_test_raw)

# Turn labels into numbers and apply One-Hot Encoding
encoder = LabelBinarizer()
encoder.fit(y_train_val_raw)
Y_train_val = encoder.transform(y_train_val_raw)
Y_test = encoder.transform(y_test_raw)

# Change to float32, so it can be multiplied against the features in TensorFlow, which are float32
Y_train_val = Y_train_val.astype(np.float32)
Y_test = Y_test.astype(np.float32)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val,
    Y_train_val,
    test_size=0.05,
    random_state=832289)

n = 784
k = 10

X = tf.placeholder(tf.float32, [None, 28*28])
Y = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.truncated_normal((n, k)))
b = tf.Variable(tf.zeros(k))
logits = tf.add(tf.matmul(X, W), b)
J = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
is_correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(J)
init = tf.global_variables_initializer()

train_feed_dict = {X: X_train, Y: Y_train}
valid_feed_dict = {X: X_val, Y: Y_val}
test_feed_dict = {X: X_test, Y: Y_test}

test_accuracy = 0.0

with tf.Session() as session:
    session.run(init)
    batch_count = int(math.ceil(len(X_train) / batch_size))

    for epoch_i in range(epochs):

        # Progress bar
        batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

        # The training cycle
        for batch_i in batches_pbar:
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size
            X_train_batch = X_train[batch_start:batch_start + batch_size]
            Y_train_batch = Y_train[batch_start:batch_start + batch_size]

            # Run optimizer
            _ = session.run(optimizer, feed_dict={X: X_train_batch, Y: Y_train_batch})

        # Check accuracy against Test data
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
print('Nice Job! Test Accuracy is {}'.format(test_accuracy))
