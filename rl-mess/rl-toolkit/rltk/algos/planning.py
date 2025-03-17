import random
from copy import deepcopy

from rltk.core.abstract.discrete_mdp import DiscreteMDP
from rltk.core.abstract.policy import Policy
from rltk.core.abstract.state_values import StateValues
from rltk import Kit, RlMetrics


def evaluate_policy(mdp: DiscreteMDP, pi: Policy, gamma=0.9) -> StateValues:
    kit = Kit.instance()
    rl_metrics = RlMetrics.instance()

    svals = kit.new_state_values()
    svals_prev = kit.new_state_values()
    for s in mdp.states:
        if mdp.is_terminal(s):
            svals[s] = 0.
            svals_prev[s] = 0.
        else:
            svals[s] = random.random()
            svals_prev[s] = random.random()

    i = -1
    while not svals.are_close(svals_prev):
        i += 1
        rl_metrics.log_svals(svals)
        svals_prev = deepcopy(svals)
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
            sval = 0.
            for a in mdp.actions:
                q = 0.
                for s_ in mdp.states:
                    r = mdp.reward(s, a, s_)
                    p = mdp.trans_prob(s, a, s_)
                    q += p * (r + gamma * svals[s_])
                sval += pi.prob(a, given=s) * q
            svals[s] = sval
    return svals

