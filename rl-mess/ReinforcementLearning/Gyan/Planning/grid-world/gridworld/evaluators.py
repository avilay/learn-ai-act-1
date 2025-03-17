from collections import Counter, defaultdict, namedtuple
from typing import Mapping, Dict, List, Sequence
from copy import deepcopy
import math
import logging
from tqdm import tqdm
import numpy as np

from .policy import Policy
from .world import World, State, Action

MAX_ITERS = 1000000
NUM_EPISODES = 100000

def converged(values: Mapping[State, float], prev_values: Mapping[State, float]) -> bool:
    for s, v in values.items():
        prev_v = prev_values[s]
        if not math.isclose(v, prev_v, rel_tol=0.000001):
            return False
    return True

def evaluate_policy(pi: Policy, world: World) -> Mapping[State, float]:
    return evaluate_policy_bellman(pi, world)
    # return evaluate_policy_brute(pi, world)

def evaluate_policy_bellman(pi: Policy, world: World) -> Mapping[State, float]:
    values: Dict[State, float] = {}
    prev_values: Dict[State, float] = {}
    for s in world.states():
        values[s] = 0.
        prev_values[s] = 10.

    for i in tqdm(range(MAX_ITERS)):
        if converged(prev_values, values):
            logging.info(f'Converged in {i} iterations')
            break

        prev_values = deepcopy(values)
        i += 1
        for s in world.states():
            if world.is_terminal(s): continue
            vnew = 0.
            action_dist = pi.pmf(s)
            for a, pa in action_dist.items():
                exp_val_s_ = 0.0
                state_dist, r = world.what_if_move(s, a)
                for s_, ps in state_dist.items():
                    exp_val_s_ += ps * prev_values[s_]
                vnew += pa * (r + exp_val_s_)
            values[s] = vnew

    return values

Step = namedtuple('Step', ['s', 'a', 'r', 's_'])

def gen_episode(pi: Policy, world: World) -> Sequence[Step]:
    episode = []
    states = list(world.states())
    start_state = states[np.random.randint(len(states))]
    a = pi(start_state)
    s_, r = world.move(start_state, a)
    step = Step(s=start_state, a=a, r=r, s_=s_)
    episode.append(step)
    while not world.is_terminal(episode[-1].s_):
        s = episode[-1].s_
        a = pi(s)
        s_, r = world.move(s, a)
        step = Step(s=s, a=a, r=r, s_=s_)
        episode.append(step)
    return episode

def evaluate_policy_brute(pi: Policy, world: World) -> Mapping[State, float]:
    episodes = [gen_episode(pi, world) for _ in range(NUM_EPISODES)]
    raw_values: Dict[State, List[float]] = defaultdict(list)

    # Compute the raw values
    for episode in tqdm(episodes):
        ep_vals: Dict[State, float] = defaultdict(lambda: 0)
        for step in reversed(episode):
            ep_vals[step.s] = step.r + ep_vals[step.s_]
        for state, val in ep_vals.items():
            raw_values[state].append(val)

    values: Dict[State, float] = {}
    for state, vals in raw_values.items():
        val = sum(vals)/len(vals)
        values[state] = val
    return values
