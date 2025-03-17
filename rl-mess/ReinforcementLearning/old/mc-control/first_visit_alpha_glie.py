from collections import defaultdict
import numpy as np
from common import calc_cum_rewards, gen_episode
from tqdm import tqdm

def gen_epsilon_greedy_policy(Q, ε):
    def policy(state, action):
        action_values = Q[state]
        best_action = np.argmax(action_values)
        probs = [ε/len(action_values)] * len(action_values)
        probs[best_action] += (1 - ε)
        return probs[action]
    return policy

def learn(env, num_episodes, γ=1.0, α=0.01):
    Q = defaultdict(lambda: [0]*env.action_space.n)
    for i_episode in tqdm(range(num_episodes)):
        ε = 1/(i_episode+1)
        policy = gen_epsilon_greedy_policy(Q, ε)
        episode = gen_episode(env, policy)
        cum_rewards = calc_cum_rewards(episode, γ)
        visited = set()
        for i_step, step in enumerate(episode):
            state = step.state
            action = step.action
            if (state, action) not in visited:
                visited.add((state, action))
                Q[state][action] = Q[state][action] + α*(cum_rewards[i_step] - Q[state][action])
    return Q, gen_epsilon_greedy_policy(Q, ε)
