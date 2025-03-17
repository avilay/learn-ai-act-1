from rltk.discrete import DiscreteMDP
from rltk.core import Policy, StateValues, ActionValues


def evaluate_policy(mdp: DiscreteMDP, pi: Policy, gamma=0.9) -> StateValues:
    v_prev = StateValues.zeros()
    v = v_prev + 0.01
    while not v_prev.are_close(v):
        # v(s) = Σπ(a|s)ΣP(s_|s,a)[R(s,a,s_) + γv(s_)]
        #      = Σπ(a|s)q(s,a)
        v_prev = v
        q: ActionValues = _build_qvals(mdp, v_prev, gamma)
        for s in mdp.states:
            v[s] = sum([pi.prob(a, given=s) * q[s, a] for a in mdp.actions])
    return v


def iterate_values(mdp: DiscreteMDP) -> StateValues:
    pass

def iterate_policy(mdp: DiscreteMDP, gamma=0.9, num_iters=10) -> StateValues:
    v_prev = StateValues.random()
    v = v_prev + 0.01
    while not v_prev.are_close(v):
        v_prev = v

        # Update policy based on values
        q: ActionValues = _build_qvals(mdp, v, gamma)
        pi = Policy.greedy(q)

        # Update values based on policy
        for _ in range(num_iters):
            for s in mdp.states:
                v[s] = sum([pi.prob(a, given=s) * q[s, a] for a in mdp.actions])
            q = _build_qvals(mdp, v, gamma)

    return v


def _build_qvals(mdp: DiscreteMDP, v: StateValues, gamma: float):
    """
    q(s,a) = ΣP(s_|s,a)[R(s,a,s_) + γv(s_)]
    """
    q: ActionValues = ActionValues.zeros()
    for s in mdp.states:
        for a in mdp.actions:
            for s_ in mdp.states:
                q[s, a] += mdp.trans_prob(s, a, s_) * (mdp.reward(s, a, s_) + gamma * v[s_])
    return q
