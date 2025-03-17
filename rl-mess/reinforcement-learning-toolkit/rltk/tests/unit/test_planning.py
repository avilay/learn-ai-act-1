import math
import pytest
from rltk.planning.planning import Planning
from rltk.core.discrete_mdp import DiscreteMDP
from rltk.core.policy import Policy


class ToyMDP(DiscreteMDP):
    def __init__(self):
        self._states = {'s0', 's1', 's2'}
        self._actions = {'a0', 'a1'}
        self._rewards = {
            's1': {'a0': {'s0': +5}},
            's2': {'a1': {'s0': -1}}
        }
        self._probs = {
            ('s0', 'a0'): [('s0', 0.5), ('s2', 0.5)],
            ('s0', 'a1'): [('s2', 1.0)],
            ('s1', 'a0'): [('s0', 0.7), ('s1', 0.1), ('s2', 0.2)],
            ('s1', 'a1'): [('s1', 0.95), ('s2', 0.05)],
            ('s2', 'a0'): [('s0', 0.4), ('s1', 0.6)],
            ('s2', 'a1'): [('s0', 0.3), ('s1', 0.3), ('s2', 0.4)],
        }

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions

    def reward(self, s, a, s_):
        return self._rewards.get(s, {}).get(a, {}).get(s_, 0.)

    def dist(self, s, a):
        return self._probs[(s, a)]

    def is_terminal(self, s):
        return False


@pytest.fixture
def mdp():
    return ToyMDP()


@pytest.fixture
def optimal_policy():
    rules = {
        's0': [('a1', 1.)],
        's1': [('a0', 1.)],
        's2': [('a0', 1.)],
    }
    return Policy(rules)


def test_evaluate_policy(mdp, optimal_policy):
    planning = Planning(mdp)
    act_svals = planning.evaluate_policy(optimal_policy)
    exp_svals = {'s0': 8.03191984357723,
                 's1': 11.171970839894161, 's2': 8.924355389898883}
    for s in ['s0', 's1', 's2']:
        assert s in act_svals and math.isclose(exp_svals[s], act_svals[s])


def test_value_iteration(mdp):
    planning = Planning(mdp)
    act_opt_svals = planning.value_iteration()
    exp_opt_svals = {'s0': 8.03191984357723,
                     's1': 11.171970839894161, 's2': 8.924355389898883}
    for s in ['s0', 's1', 's2']:
        assert s in act_opt_svals and math.isclose(
            exp_opt_svals[s], act_opt_svals[s])


def test_policy_iteration(mdp):
    planning = Planning(mdp)
    act_opt_svals = planning.policy_iteration()
    exp_opt_svals = {'s0': 8.03191984357723,
                     's1': 11.171970839894161, 's2': 8.924355389898883}
    for s in ['s0', 's1', 's2']:
        assert s in act_opt_svals and math.isclose(
            exp_opt_svals[s], act_opt_svals[s])
