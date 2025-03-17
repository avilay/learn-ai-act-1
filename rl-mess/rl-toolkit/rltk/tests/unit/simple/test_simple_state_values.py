import random

from rltk.simple import SimpleStateValues


def test_getset(states):
    svals = SimpleStateValues()
    for i, s in enumerate(states):
        svals[s] = i
    for i, s in enumerate(states):
        assert i == svals[s]


def test_add(states):
    svals = SimpleStateValues()
    for i, s in enumerate(states):
        svals[s] = i
    svals_ = svals + 1
    for i, s in enumerate(states):
        assert i+1 == svals_[s]
        assert i == svals[s]


def test_iter(states):
    raw_svals = {}
    svals = SimpleStateValues()
    for s in states:
        v = random.random()
        raw_svals[s] = v
        svals[s] = v

    for s, v in svals:
        assert raw_svals[s] == v


def test_areclose(states):
    svals = SimpleStateValues()
    svals_ = SimpleStateValues()
    for i, s in enumerate(states):
        svals[s] = i
        svals_[s] = i + 0.0003
    assert svals.are_close(svals_)

    svals_['snew'] = 3.14
    assert not svals.are_close(svals_)

    svals_ = SimpleStateValues()
    for i, s in enumerate(states):
        svals_[s] = i + 0.01
    assert not svals.are_close(svals_)
