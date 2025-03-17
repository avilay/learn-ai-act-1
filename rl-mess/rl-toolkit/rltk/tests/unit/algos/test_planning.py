import math
from typing import Iterator, Tuple, List

from mlmetrics.metric import Metric

import rltk.algos.planning as planning
from rltk.core import State
from rltk.core import Policy, DiscreteMDP, StateValues
import numpy as np
import pytest
from rltk import Kit


class UTMetric(Metric):
    def __init__(self, name, labels):
        pass

    def log(self, **kwargs) -> None:
        pass

    def start_snapshot(self) -> None:
        pass

    def stop_snapshot(self) -> None:
        pass

    def logs(self, start=-1, end=-1, **filters) -> Iterator[Tuple]:
        pass

    def snapshots(self, start=-1, end=-1, **filters) -> Iterator[List[Tuple]]:
        pass

    def range(self, start=-1, end=-1, **filters) -> Tuple[float, float]:
        pass

    def close(self) -> None:
        pass


class UTPolicy(Policy):
    def __init__(self):
        self._dist = {
            "s0": {"a0": 0.0, "a1": 1.0},
            "s1": {"a0": 1.0, "a1": 0.0},
            "s2": {"a0": 1.0, "a1": 0.0},
        }

    def prob(self, action, given):
        return self._dist[given][action]

    def action(self, state):
        actions = []
        probs = []
        for action, prob in self._dist[state].items():
            actions.append(action)
            probs.append(prob)
        return np.random.choice(actions, p=probs)

    @classmethod
    def deterministic(cls, rules):
        raise NotImplementedError()


class UTMDP(DiscreteMDP):
    def __init__(self):
        self.probs = {
            ("s0", "a0"): {"s0": 0.5, "s2": 0.5},
            ("s0", "a1"): {"s2": 1.0},
            ("s1", "a0"): {"s0": 0.7, "s1": 0.1, "s2": 0.2},
            ("s1", "a1"): {"s1": 0.95, "s2": 0.05},
            ("s2", "a0"): {"s0": 0.4, "s1": 0.6},
            ("s2", "a1"): {"s0": 0.3, "s1": 0.3, "s2": 0.4},
        }
        self.terminal_states = []

    def reward(self, s, a, s_):
        if s == "s1" and a == "a0" and s_ == "s0":
            return 5
        elif s == "s2" and a == "a1" and s_ == "s0":
            return -1
        else:
            return 0

    def trans_prob(self, s, a, s_):
        return self.probs.get((s, a), {}).get(s_, 0.0)

    def is_terminal(self, s):
        return s in self.terminal_states

    @property
    def states(self):
        return ["s0", "s1", "s2"]

    @property
    def actions(self):
        return ["a0", "a1"]


class UTStateValues(StateValues):
    def __init__(self):
        self._svals = {"s0": float("nan"), "s1": float("nan"), "s2": float("nan")}

    def __getitem__(self, state: State) -> float:
        return self._svals[state]

    def __setitem__(self, state: State, value: float) -> None:
        self._svals[state] = value

    def __len__(self) -> int:
        return len(self._svals)

    def __add__(self, value: float) -> "StateValues":
        svals = {}
        for s, v in self._svals:
            svals[s] = v + value
        obj = UTStateValues()
        obj._svals = svals
        return obj

    def __iter__(self) -> Iterator[Tuple[State, float]]:
        for s, v in self._svals:
            yield s, v

    def are_close(self, other: "StateValues") -> bool:
        if (len(self) == 0 and len(other) == 0) or len(self) != len(other):
            return False
        for s in self._svals.keys():
            v1 = self._svals[s]
            v2 = other[s]
            if not math.isclose(v1, v2, rel_tol=0.001, abs_tol=0.001):
                return False
        return True


@pytest.fixture
def mdp():
    return UTMDP()


@pytest.fixture
def policy():
    return UTPolicy()


@pytest.fixture
def exp_svals():
    return {"s0": 8.006988800351264, "s1": 11.14839253622131, "s2": 8.902680033071245}


def test_evaluate_policy(mdp, policy, exp_svals):
    kit = Kit.instance()
    kit.state_values = UTStateValues
    kit.metric = UTMetric
    act_svals = planning.evaluate_policy(mdp, policy)
    print(act_svals)
    assert 3 == len(act_svals)
    for s in exp_svals.keys():
        assert math.isclose(exp_svals[s], act_svals[s], rel_tol=0.001, abs_tol=0.001)
