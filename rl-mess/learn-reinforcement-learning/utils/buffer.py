from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class Buffer(ABC):
    def __init__(self, env, capacity, avg_over):
        if capacity < 0:
            raise ValueError("Capacity must be positive!")
        if avg_over < 0:
            raise ValueError("Scores can be averaged over positive number steps!")
        self._avg_over = avg_over
        self._env = env
        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)
        self._scores = deque(maxlen=avg_over)

    @abstractmethod
    def replinish(self, policy, fraction):
        pass

    def sample(self, size):
        if size > len(self._buffer):
            raise ValueError("Not enough samples in buffer!")
        idxs = np.random.randint(len(self._buffer), size=size)
        batch = []
        for idx in idxs:
            batch.append(self._buffer[idx])
        return batch

    @property
    def avg_score(self):
        if not self._scores:
            raise RuntimeError("Not enough scores in the buffer!")
        return np.mean(self._scores)

    @property
    def latest_score(self):
        if not self._scores:
            raise RuntimeError("Not enough scores in the buffer!")
        return self._scores[-1]
