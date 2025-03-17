from typing import Any, Callable, List

import numpy as np
from gym.core import Env

from .episode import Episode
from .step import Step

Policy = Callable[[int, int], float]
ValueFunc = Callable[[Any], float]


def build(env: Env, π: Policy, num_steps: int) -> List[Episode]:
    episodes = []
    episode = Episode()
    s = env.reset()
    for _ in range(num_steps):
        action_probs = np.array([π(a, s) for a in range(env.action_space.n)])
        a = np.random.choice(list(range(env.action_space.n)), p=action_probs)
        s_, r, done, _ = env.step(a)
        episode.append(Step(state=s, action=a, reward=r, next_state=s_))
        if done:
            episodes.append(episode)
            episode = Episode()
            s = env.reset()
        else:
            s = s_

    # throw away the last episode if it is not complete and NUM_STEPS has been reached
    return episodes
