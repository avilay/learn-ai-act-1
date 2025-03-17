from dataclasses import dataclass

import numpy as np
import tensorflow as tf


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray


class Agent:
    def __init__(self, env, batch_size):
        self._env = env
        self._batch_size = batch_size

    def gen_batch(self, policy, dbg=False):
        states = []
        actions = []
        returns = []
        scores = []
        while len(states) < self._batch_size:
            ep_states = []
            ep_actions = []
            state = self._env.reset()
            done = False
            score = 0
            while not done:
                state_batch_of_one = np.expand_dims(state, axis=0)
                action_probs = policy(state_batch_of_one)
                # action = tf.random.categorical(tf.math.log(action_probs), 1).numpy().squeeze()
                action_probs = action_probs.numpy().squeeze()
                action = np.random.choice(np.arange(self._env.action_space.n), p=action_probs)
                next_state, reward, done, _ = self._env.step(action)
                score += reward
                ep_states.append(state)
                ep_actions.append(action)
                state = next_state
            ep_returns = [score] * len(ep_states)
            states += ep_states
            actions += ep_actions
            returns += ep_returns
            scores.append(score)
        batch = Batch(
            states=np.array(states[: self._batch_size], dtype=np.float32),
            actions=np.array(actions[: self._batch_size], dtype=np.int),
            returns=np.array(returns[: self._batch_size], dtype=np.float32),
        )
        return batch, scores
