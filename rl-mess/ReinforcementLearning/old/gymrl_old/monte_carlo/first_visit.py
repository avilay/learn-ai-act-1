from collections import defaultdict
from typing import Mapping, Any, Dict, List, Set, Tuple, Sequence, MutableMapping
import numpy as np
import gym
from tqdm import tqdm

from ..common import Step, gen_episode, PolicyFunction, calc_cum_rewards

def learn(env: gym.Env, num_episodes: int, policy: PolicyFunction, γ=1.0) -> Mapping[Any, np.ndarray]:
    # Total rewards for this state-action pair across episodes
    returns_sum: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space.n))

    # Total number of episodes which featured this state-action pair
    N: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space.n))

    episodes: List[Sequence[Step]] = [gen_episode(env, policy) for _ in range(num_episodes)]
    for episode in tqdm(episodes):
        cum_rewards = calc_cum_rewards(episode, γ)

        # Keep track of state-action pairs that have already occured in this episode
        visited: Set[Tuple[Any, int]] = set()

        for i_step, step in enumerate(episode):
            state = step.state
            action = step.action
            if (state, action) not in visited:
                visited.add((state, action))
                returns_sum[state][action] += cum_rewards[i_step]
                N[state][action] += 1

    Q: MutableMapping[Any, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space.n))
    for state in N.keys():
        rewards = np.array(returns_sum[state], dtype=np.float)
        num_times = np.array(N[state], dtype=np.float)
        num_times[num_times == 0] += 1  # Replace 0s with 1s to avoid divide-by-zero error
        Q[state] = rewards / num_times

    return Q
