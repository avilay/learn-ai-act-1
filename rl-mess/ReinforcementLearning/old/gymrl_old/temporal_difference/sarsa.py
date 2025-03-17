# from collections import defaultdict
import gym
# import numpy as np
# from typing import Dict, Any, Mapping, Sequence
from ..common.q_table import QTable
from ..common.policy import Policy


def learn(env: gym.Env, num_episodes: int, α: float, γ: float=1.0) -> QTable:
    qtab = QTable(env.action_space.n)
    for i_episode in range(1, num_episodes+1):
        ε = 1/i_episode
        policy: Policy = qtab.epsilon_greedy_policy(ε)
        state = env.reset()
        action = policy.action(state)
        next_state, next_reward, done, _ = env.step(action)
        while not done:
            next_action = policy.action(next_state)
            qtab[state, action] += α * (next_reward + γ * qtab[next_state, next_action] - qtab[state, action])

            policy = qtab.epsilon_greedy_policy(ε)

            state = next_state
            action = policy.action(state)
            next_state, next_reward, done, _ = env.step(action)
    return qtab

# def learn_old(env: gym.Env, num_episodes: int, α: float, γ: float=1.0) -> Mapping[Any, Sequence[float]]:
#     num_actions = env.action_space.n
#     Q: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(num_actions))
#     for i_episode in range(1, num_episodes+1):
#         # Start the episode
#         ε = 1/i_episode
#         policy = gen_epsilon_greedy_policy(Q, ε)
#         state = env.reset()
#         action = np.random.choice(np.arange(num_actions), policy(state))
#         next_state, next_reward, done, info = env.step(action)
#         while not done:
#             # Update the Q-table based on the current policy
#             next_action = np.random.choice(np.arange(num_actions), policy(next_state))
#             Q[state][action] += α * (next_reward + γ * Q[next_state][next_action] - Q[state][action])

#             # Update the policy based on the updated Q-table
#             policy = gen_epsilon_greedy_policy(Q, ε)

#             # Make the next move based on the updated policy
#             state = next_state
#             action = np.random.choice(np.arange(num_actions), policy(state))
#             next_state, next_reward, done, info = env.step(action)
#     return Q
