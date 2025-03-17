import itertools
import random

import pytest

from rltk.core.abstract.state_values import StateValues
from rltk.core.abstract.action_values import ActionValues
import math
from rltk import Kit


class TestStateValues(StateValues):
    def __init__(self):
        self._svals = {
            's0': float('nan'),
            's1': float('nan'),
            's2': float('nan')
        }

    def __getitem__(self, state):
        return self._svals[state]

    def __setitem__(self, state, value):
        self._svals[state] = value

    def __len__(self):
        return len(self._svals)

    def __add__(self, value):
        newsvals = TestStateValues()
        for s in self._svals:
            newsvals[s] = self._svals[s] + value
        return newsvals

    def __repr__(self):
        return f'<StateValues(s0={self._svals["s0"]} s1={self._svals["s1"]} s2={self._svals["s2"]})>'

    def are_close(self, other: 'StateValues') -> bool:
        if len(self) != len(other):
            return False
        for s in self._svals.keys():
            v1 = self._svals[s]
            v2 = other[s]
            if not math.isclose(v1, v2, rel_tol=0.001, abs_tol=0.001):
                return False
        return True

    @classmethod
    def zeros(cls):
        obj = cls()
        for state in ['s0', 's1', 's2']:
            obj._svals[state] = 0.0
        return obj

    @classmethod
    def random(cls):
        obj = cls()
        for state in ['s0', 's1', 's2']:
            obj._svals[state] = random.uniform(0.1, 0.99)
        return obj


class TestActionValues(ActionValues):
    def __init__(self):
        self._qvals = {
            's0': {'a0': float('nan'), 'a1': float('nan')},
            's1': {'a0': float('nan'), 'a1': float('nan')},
            's2': {'a0': float('nan'), 'a1': float('nan')}
        }

    def __getitem__(self, key):
        s, a = key
        return self._qvals[s][a]

    def __setitem__(self, key, value):
        s, a = key
        self._qvals[s][a] = value

    def __len__(self):
        return 6

    def __add__(self, value):
        newqvals = TestActionValues()
        for s, a in itertools.product(['s0', 's1', 's2'], ['a0', 'a1']):
            newqvals[s,a] = self._qvals[s][a] + value
        return newqvals

    def are_close(self, other):
        if len(self) != len(other):
            return False
        for s, avals in self._qvals.items():
            for a, v1 in avals.items():
                v2 = other[s, a]
                if not math.isclose(v1, v2, rel_tol=0.001, abs_tol=0.001):
                    return False
        return True

    @classmethod
    def zeros(cls):
        raise NotImplemented()

    @classmethod
    def random(cls):
        raise NotImplemented()


Kit.instance().action_values = ActionValues


@pytest.fixture
def state_values():
    svals = TestStateValues()
    svals['s0'] = 8.006657803340513
    svals['s1'] = 11.148268441366525
    svals['s2'] = 8.902461767540508
    return svals
