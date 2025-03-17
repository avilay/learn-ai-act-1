import random

from rltk.simple import SimpleActionValues
import itertools


def test_getset(states, actions):
    qvals = SimpleActionValues()
    for i, (s, a) in enumerate(itertools.product(states, actions)):
        qvals[s,a] = i
    for i, (s, a) in enumerate(itertools.product(states, actions)):
        assert i == qvals[s,a]


def test_iter(states, actions):
    raw_qvals = {}
    qvals = SimpleActionValues()
    for s, a in itertools.product(states, actions):
        val = random.random()
        raw_qvals[(s,a)] = val
        qvals[s,a] = val

    for s, a, v in qvals:
        assert raw_qvals[s,a] == v


def test_add(states, actions):
    qvals = SimpleActionValues()
    for i, (s, a) in enumerate(itertools.product(states, actions)):
        qvals[s,a] = i
    qvals_ = qvals + 1
    for i, (s, a) in enumerate(itertools.product(states, actions)):
        assert i+1 == qvals_[s,a]
        assert i == qvals[s,a]  # ensure that the original obj has not changed


def test_are_close(states, actions):
    qvals = SimpleActionValues()
    qvals_ = SimpleActionValues()
    for i, (s, a) in enumerate(itertools.product(states, actions)):
        qvals[s,a] = i
        qvals_[s,a] = i + 0.0003
    act_are_close = qvals.are_close(qvals_)
    assert act_are_close, f'Actual value is {act_are_close}'

    qvals_['snew', 'anew'] = 3.14
    assert not qvals.are_close(qvals_)

    qvals_ = SimpleActionValues()
    for i, (s, a) in enumerate(itertools.product(states, actions)):
        qvals_[s,a] = i + 10
    assert not qvals.are_close(qvals_)
