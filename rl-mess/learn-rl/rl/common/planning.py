from collections import defaultdict

import numpy as np
from gym.envs.toy_text.discrete import DiscreteEnv

from .typedefs import Mdp, Policy


def build_mdp(env: DiscreteEnv) -> Mdp:
    # P[s][a] = [(probability, nextstate, reward, done), ...]
    # (s, a, s_) => R(s, a, s_)
    rewards = defaultdict(float)
    # (s, a, s_) => P(s_|s, a)
    probs = defaultdict(float)

    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            psa = env.P[s][a]
            for p, s_, r, _ in psa:
                rewards[(s, a, s_)] += r
                probs[(s, a, s_)] += p
    return rewards, probs


def calc_state_vals(env: DiscreteEnv, π: Policy, γ=0.9, max_iters=1000) -> np.ndarray:
    v_new = np.zeros(env.observation_space.n)
    v = np.random.random(env.observation_space.n)
    rewards, probs = build_mdp(env)
    for _ in range(max_iters):
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                q = 0.0
                for s_ in range(env.observation_space.n):
                    r = rewards[(s, a, s_)]
                    p = probs[(s, a, s_)]
                    q += p * (r + γ * v[s_])
                v_new[s] += π(a, s) * q
        if np.allclose(v, v_new):
            break
        v = v_new
        v_new = np.zeros(env.observation_space.n)
    return v
