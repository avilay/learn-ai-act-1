import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class PolicyNet(Model):
    def __init__(self, n_actions):
        super().__init__()
        self.fc1 = Dense(64, activation="tanh", dtype=tf.float32)
        self.fc2 = Dense(64, activation="tanh", dtype=tf.float32)
        self.fc3 = Dense(n_actions, activation="softmax", dtype=tf.float32)

    def call(self, states):
        return self.fc3(self.fc2(self.fc1(states)))
