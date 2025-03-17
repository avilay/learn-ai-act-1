from .q_table import QTable
from .epsilon_greedy_policy import EpsilonGreedyPolicy
import gym
from tqdm import tqdm


def learn(env: gym.Env, num_episodes: int, α: float, γ: float=1.0) -> QTable:
    qtab = QTable(env.action_space.n)
    for i_episode in tqdm(range(1, num_episodes+1)):
        ε = 1/i_episode
        policy = EpsilonGreedyPolicy(qtab, ε)
        state = env.reset()
        action = policy.action(state)
        next_state, next_reward, done, _ = env.step(action)
        while not done:
            exp_reward = 0.
            for a in range(env.action_space.n):
                exp_reward += policy.prob(next_state, a) * qtab[next_state, a]
            qtab[state, action] += α * (next_reward + γ * exp_reward - qtab[state, action])
            policy = EpsilonGreedyPolicy(qtab, ε)
            state = next_state
            action = policy.action(state)
            next_state, next_reward, done, _ = env.step(action)
    return qtab
