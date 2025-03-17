from rltk.core import DiscreteMDP, Policy
from rltk.planning import Planning, SqlitePlanningMetricWriter
import rltk.metrics as metrics
from pprint import pprint
import logging

logging.basicConfig(level=logging.DEBUG)


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


rules = {
    's0': [('a1', 1.)],
    's1': [('a0', 1.)],
    's2': [('a0', 1.)],
}
pi = Policy(rules)

# metrics.get_logger().set_writer(metrics.ConsoleWriter())
writer = SqlitePlanningMetricWriter('/Users/avilay/temp/rltk.db')
metrics.set_writer(writer)
planning = Planning(ToyMDP())
# svals = planning.evaluate_policy(pi)
svals = planning.value_iteration()
# svals = planning.policy_iteration()
pprint(svals)
writer.close()
