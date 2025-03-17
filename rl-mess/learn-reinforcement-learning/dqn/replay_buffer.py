import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, env, capacity):
        self._env = env
        self._capacity = capacity
        self._buf = deque(maxlen=capacity)
        self._state = self._env.reset()
        self._scores = []
        self._score = 0

    def _gen_eps_policy(self, Q, eps):
        def policy(state):
            if random.random() > eps:
                state = np.expand_dims(state, axis=0)
                action = np.argmax(Q(state))
            else:
                action = random.choice(np.arange(self._env.action_space.n))
            return action

        return policy

    @property
    def avg_score(self):
        if self._scores:
            return np.mean(self._scores[-20:]) if len(self._scores) > 20 else np.mean(self._scores)
        else:
            return 0

    @property
    def last_score(self):
        return self._scores[-1] if self._scores else 0.0

    def replinish(self, percent, Q, eps):
        num_steps = int(self._capacity * percent)
        policy = self._gen_eps_policy(Q, eps)
        for _ in range(num_steps):
            action = policy(self._state)
            next_state, reward, done, _ = self._env.step(action)
            self._score += reward
            self._buf.append((self._state, action, reward, next_state, done))
            if done:
                self._state = self._env.reset()
                self._scores.append(self._score)
                self._score = 0
            else:
                self._state = next_state

    def sample(self, size):
        idxs = np.random.randint(len(self._buf), size=size)
        steps = [self._buf[idx] for idx in idxs]
        states = []
        actions = []
        rewards = []
        next_steps = []
        dones = []
        for step in steps:
            states.append(step[0])
            actions.append(step[1])
            rewards.append(step[2])
            next_steps.append(step[3])
            dones.append(step[4])
        return Batch(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_steps),
            np.array(dones, np.int),
        )
