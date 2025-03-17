from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class DQN(Model):
    def __init__(self, n_actions):
        super().__init__()
        self.fc1 = Dense(64, activation="relu")
        self.fc2 = Dense(64, activation="relu")
        self.fc3 = Dense(n_actions)

    def call(self, states):
        x = self.fc1(states)
        x = self.fc2(x)
        return self.fc3(x)
