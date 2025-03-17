from collections import defaultdict
import numpy as np
from common import STICK, HIT, calc_cum_rewards, gen_episode
from tqdm import tqdm

def policy(state, action):
    player_hand = state[0]
    if player_hand > 18:
        return 0.8 if action == STICK else 0.2
    else:
        return 0.8 if action == HIT else 0.2

def learn(env, num_episodes, γ=1.0):
    # Dictionary keyed by state with a numpy array as value
    # returns_sum[STATE][ACTION] = Total rewards for this state-action pair across episodes
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))

    # N[STATE][ACTION] = Total number of episodes that had this state-action pair
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    episodes = [gen_episode(env, policy) for _ in range(num_episodes)]

    for episode in tqdm(episodes):
        cum_rewards = calc_cum_rewards(episode, γ)

        # Keep track of state-action pairs that have already occurred in this episode
        visited = set()

        for i_step, step in enumerate(episode):
            state = step.state
            action = step.action
            if (state, action) not in visited:
                visited.add((state, action))
                returns_sum[state][action] += cum_rewards[i_step]
                N[state][action] += 1

    # Dictionary keyed by state with a list as value
    # Q[STATE][ACTION] = avg. reward for this state-action pair across episodes
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for state in N.keys():
        rewards = np.array(returns_sum[state], dtype=np.float)
        num_times = np.array(N[state], dtype=np.float)
        num_times[num_times == 0] += 1  # Replace 0s with 1s to avoid divide by 0 error
        Q[state] = rewards / num_times

    return Q, policy
