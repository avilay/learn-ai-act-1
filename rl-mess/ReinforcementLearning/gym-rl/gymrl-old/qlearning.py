import numpy as np
from tqdm import tqdm

import gym
import gym.spaces

from .epsilon_greedy_policy import EpsilonGreedyPolicy
from .q_table import QTable


def learn(env: gym.Env, num_epsiodes: int, α: float, γ: float=1.0) -> QTable:
    qtab = QTable(env.action_space.n)
    for i_episode in tqdm(range(1, num_epsiodes+1)):
        # ε = max(1/i_episode, 0.01)
        ε = 1/i_episode
        policy = EpsilonGreedyPolicy(qtab, ε)
        state = env.reset()
        action = policy.action(state)
        next_state, next_reward, done, _ = env.step(action)
        while not done:
            qtab[state, action] += α * (next_reward + γ * np.max(np.array(qtab.all_values(next_state))) - qtab[state, action])
            policy = EpsilonGreedyPolicy(qtab, ε)
            state = next_state
            action = policy.action(state)
            next_state, next_reward, done, _ = env.step(action)
    return qtab
