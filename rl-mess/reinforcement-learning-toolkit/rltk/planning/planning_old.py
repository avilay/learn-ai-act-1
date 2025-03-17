dfrom collections import defaultdict
from copy import copy
from typing import Dict, List, Mapping, Tuple
import logging

import numpy as np

from rltk.mathext import has_converged
import rltk.metrics as metrics
from rltk.core.discrete_mdp import DiscreteMDP
from rltk.core.policy import Policy
from rltk.core.types import Action, State

from .planning_metric import PlanningMetric

logger = logging.getLogger(__name__)


class Planning:
    def __init__(self, mdp: DiscreteMDP, gamma=0.9) -> None:
        self._mdp = mdp
        self._gamma = gamma

    def evaluate_policy(self, policy: Policy) -> Dict[State, float]:
        """
        Given an MDP and a policy, finds the state values under that policy.
        """
        svals: Dict[State, float] = {s: 0.1 for s in self._mdp.states}
        new_svals: Dict[State, float] = {s: 0. for s in self._mdp.states}
        i = 0
        while not has_converged(svals, new_svals):
            metric = PlanningMetric(context_id='evalulate_policy', iter_num=i)
            svals = copy(new_svals)
            for s in self._mdp.states:
                if self._mdp.is_terminal(s):
                    continue
                qvals = {a: self._calc_qval(svals, s, a) for a in self._mdp.actions}
                metric.qvals[s] = qvals
                pi = policy.dist(s)
                metric.pidist[s] = pi
                new_svals[s] = sum([qvals[a]*pi[a] for a in self._mdp.actions])
            metric.svals = new_svals
            metrics.log(metric)
            i += 1
        logger.info(f'evaluate_policy completed in {i} iterations.')
        return svals

    def value_iteration(self) -> Dict[State, float]:
        """
        Given an MDP find the optimal state values.
        """
        svals = {s: 0.1 for s in self._mdp.states}
        new_svals = {s: 0. for s in self._mdp.states}
        i = 0
        while not has_converged(svals, new_svals):
            metric = PlanningMetric(context_id='value_iteration', iter_num=i)
            svals = copy(new_svals)
            for s in self._mdp.states:
                if self._mdp.is_terminal(s):
                    continue
                qvals = {a: self._calc_qval(svals, s, a) for a in self._mdp.actions}
                metric.qvals[s] = qvals
                new_svals[s] = max(qvals.values())
            metric.svals = new_svals
            metrics.log(metric)
            i += 1
        logger.info(f'value_iteration completed in {i} iterations.')
        return svals

    # def policy_iteration(self) -> Dict[State, float]:
    #     """
    #     Given an MDP find the optimal state values
    #     """
    #     pi: Policy = self.gen_random_policy()
    #     new_svals: Dict[State, float] = self.evaluate_policy(pi)
    #     svals: Dict[State, float] = {s: new_svals[s] - 1. for s in self._mdp.states}
    #     i = 0
    #     while not has_converged(svals, new_svals):
    #         metric = PlanningMetric(context_id='policy_iteration', iter_num=i)
    #         svals = copy(new_svals)
    #         pi = self.gen_greedy_policy(svals)
    #         new_svals = self.evaluate_policy(pi)
    #         metric.svals = new_svals
    #         metric.pidist = {s: pi.dist(s) for s in self._mdp.states}
    #         metrics.log(metric)
    #         i += 1
    #     logger.info(f'policy_iteration completed in {i} iterations.')
    #     return svals

    def policy_iteration(self) -> Dict[State, float]:
        svals: Dict[State, float] = {s: np.random.random() for s in self._mpd.states}
        pi: Policy = self.gen_greedy_policy(svals)
        while True:
           for s in self._mdp.states:
               qvals = {a: self._calc_qval(svals, s, a) for a in self._mdp.actions}
               svals[s] = max(qvals)



    def _calc_qval(self, svals: Mapping[State, float], s: State, a: Action) -> float:
        tot = 0.
        for s_, p in self._mdp.dist(s, a):
            r = self._mdp.reward(s, a, s_)
            v_ = svals[s_]
            tot += p*(r + self._gamma*v_)
        return tot

    def gen_random_policy(self) -> Policy:
        rules = {}
        actions = list(self._mdp.actions)
        for s in self._mdp.states:
            a = np.random.choice(actions)
            rules[s] = [(a, 1.)]
        return Policy(rules)

    def gen_greedy_policy(self, state_vals: Mapping[State, float]) -> Policy:
        rules: Dict[State, List[Tuple[Action, float]]] = defaultdict(list)
        for s in self._mdp.states:
            actions = []
            qvals = []
            for a in self._mdp.actions:
                actions.append(a)
                qvals.append(self._calc_qval(state_vals, s, a))
            best_action = actions[np.argmax(qvals)]
            rules[s].append((best_action, 1.))
        return Policy(rules)
