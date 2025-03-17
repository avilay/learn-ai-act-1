from rltk.core import State, Action
import pytest


State.register(str)
Action.register(str)


@pytest.fixture
def states():
    return ['s0', 's1', 's2']


@pytest.fixture
def actions():
    return ['a0', 'a1']
