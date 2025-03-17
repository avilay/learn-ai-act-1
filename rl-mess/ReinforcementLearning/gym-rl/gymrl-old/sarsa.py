from .q_table import QTable
from .epsilon_greedy_policy import EpsilonGreedyPolicy
import gym
from tqdm import tqdm


def learn(env: gym.Env, num_epsiodes: int, α: float, γ: float=1.0) -> QTable:
    qtab = QTable(env.action_space.n)
    for i_episode in tqdm(range(1, num_epsiodes+1)):
        ε = 1/i_episode
        policy = EpsilonGreedyPolicy(qtab, ε)
        state = env.reset()
        action = policy.action(state)
        next_state, next_reward, done, _ = env.step(action)
        # print(qtab)
        # print(f'state={state} action={action} reward={next_reward} next_state={next_state}')
        while not done:
            next_action = policy.action(next_state)

            # print(f'Updated qtab[{state}, {action}]')
            qtab[state, action] += α * (next_reward + γ * qtab[next_state, next_action] - qtab[state, action])
            # print(qtab)

            policy = EpsilonGreedyPolicy(qtab, ε)

            state = next_state
            action = policy.action(state)
            # input('Press ENTER to continue to next time step')
            next_state, next_reward, done, _ = env.step(action)
            # print(f'state={state} action={action} reward={next_reward} next_state={next_state}')
    return qtab
