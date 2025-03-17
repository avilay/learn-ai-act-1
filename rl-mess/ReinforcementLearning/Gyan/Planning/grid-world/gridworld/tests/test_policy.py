from collections import Counter
import pytest
from pytest import fixture
from ..policy import Policy
from ..world import Action, State

def test_stochastic():
    def pmf(s):
        return {
            Action.UP: 0.25,
            Action.DOWN: 0.25,
            Action.LEFT: 0.25,
            Action.RIGHT: 0.25
        }

    policy = Policy(pmf)
    actions = []
    s = State(row=0, col=0)
    for _ in range(1000):
        a = policy(s)
        actions.append(a)
    counts = Counter(actions)
    assert 200 <= counts[Action.UP] <= 300
    assert 200 <= counts[Action.DOWN] <= 300
    assert 200 <= counts[Action.LEFT] <= 300
    assert 200 <= counts[Action.RIGHT] <= 300

def test_deterministic():
    def pmf(s):
        if s == State(row=0, col=0):
            return {
                Action.UP: 1.0,
                Action.DOWN: 0.0,
                Action.LEFT: 0.0,
                Action.RIGHT: 0.0
            }
        elif s == State(row=2, col=2):
            return {
                Action.UP: 0.0,
                Action.DOWN: 1.0,
                Action.LEFT: 0.0,
                Action.RIGHT: 0.0
            }

    policy = Policy(pmf)

    actions = []
    s = State(row=0, col=0)
    for _ in range(100):
        a = policy(s)
        actions.append(a)
    counts = Counter(actions)
    assert 100 == counts[Action.UP]

    actions = []
    for _ in range(100):
        s = State(row=2, col=2)
        a = policy(s)
        actions.append(a)
    counts = Counter(actions)
    assert 100 == counts[Action.DOWN]
