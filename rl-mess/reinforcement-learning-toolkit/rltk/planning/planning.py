import functools
from copy import copy
from typing import Dict, List, Mapping, Tuple

from rltk.core.discrete_mdp import DiscreteMDP
from rltk.core.types import State, Action, Reward
from rltk.core.policy import Policy
from rltk.mathext import has_converged
from rltk.core.qvals import QVals

class Planning:
    def __init__(self, mdp: DiscreteMDP, gamma=0.9) -> None:
        self._mdp = mdp
        self._gamma = gamma

    def evaluate_policy(self, pi: Policy) -> Dict[State, float]:
        v_prev: Dict[State, float] = {s: 0. for s in self._mdp.states}
        v: Dict[State, float] = {s: v_prev[s]+0.1 for s in self._mdp.states}
        while not has_converged(v_prev, v):
            v_prev = copy(v)
            q: QVals = self._build_qvals(v_prev)
            for s in self._mdp.states:
                v[s] = sum([pi(s, a) * q[s, a] for a in self._mdp.actions])
        return v

    def _build_qvals(self, v):
        q = QVals(self._mdp.states, self._mdp.actions, random=False)
        for s in self._mdp.states:
            for a in self._mdp.actions:
                p = functools.partial(self._mdp, s, a)
                r = functools.partial(self._mdp, s, a)
                q[s, a] = sum([p(s_) * (r(s_) + self._gamma * v[s_]) for s_ in self._mdp.states])
        return q
