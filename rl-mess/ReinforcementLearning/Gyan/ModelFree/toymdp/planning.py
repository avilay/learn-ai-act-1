from typing import Dict, List, Tuple
from collections import defaultdict
from copy import copy
import math
from functools import partial
from .types import State, Action
import numpy as np
from .policy import Policy

def evaluate_policy(mdp, policy, gamma=0.9):
    svals = {s: 0.1 for s in mdp.states}
    qval = partial(calc_qval, mdp, gamma)
    new_svals = {s: 0. for s in mdp.states}
    while not has_converged(svals, new_svals):
        svals = copy(new_svals)
        for s in mdp.states:
            if mdp.is_terminal(s): continue
            qvals = {a: qval(svals, s, a) for a in mdp.actions}
            pi = policy.dist(s)
            new_svals[s] = sum([qvals[a]*pi[a] for a in mdp.actions])
    return svals

def gen_greedy_policy(mdp, state_vals, gamma=0.9):
    qval = partial(calc_qval, mdp, gamma, state_vals)
    rules: Dict[State, List[Tuple[Action, float]]] = defaultdict(list)
    for s in mdp.states:
        actions = []
        qvals = []
        for a in mdp.actions:
            actions.append(a)
            qvals.append(qval(s, a))
        best_action = actions[np.argmax(qvals)]
        rules[s].append((best_action, 1.))
    return Policy(rules)

def has_converged(hsh1, hsh2):
    for k in hsh1.keys():
        v1 = hsh1[k]
        v2 = hsh2[k]
        if not math.isclose(v1, v2):
            return False
    return True

def calc_qval(mdp, gamma, svals, s, a):
    tot = 0.
    for s_, p in mdp.dist(s, a):
        r = mdp.reward(s, a, s_)
        v_ = svals[s_]
        tot += p*(r + gamma*v_)
    return tot

def value_iteration(mdp, gamma=0.9):
    svals = {s: 0.1 for s in mdp.states}
    qval = partial(calc_qval, mdp, gamma)
    new_svals = {s: 0. for s in mdp.states}
    while not has_converged(svals, new_svals):
        svals = copy(new_svals)
        for s in mdp.states:
            if mdp.is_terminal(s): continue
            qvals = [qval(svals, s, a) for a in mdp.actions]
            new_svals[s] = max(qvals)
    return svals
