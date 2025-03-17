import numpy as np
from pytest import fixture  # NOQA
from ..greedy_policy import GreedyPolicy  # NOQA
from ..q_table import QTable  # NOQA


@fixture
def uniq_qtab():
    qt = QTable(3)

    qt['s0', 0] = 10.
    qt['s0', 1] = 20.
    qt['s0', 2] = 5.

    qt['s1', 0] = 15.
    qt['s1', 1] = 3.
    qt['s1', 2] = 5.

    qt['s2', 0] = 1.
    qt['s2', 1] = 2.
    qt['s2', 2] = 5.

    return qt


@fixture
def dup_qtab():
    qt = QTable(3)

    qt['s0', 0] = 10.
    qt['s0', 1] = 20.
    qt['s0', 2] = 20.

    qt['s1', 0] = 15.
    qt['s1', 1] = 3.
    qt['s1', 2] = 5.

    qt['s2', 0] = 5.
    qt['s2', 1] = 5.
    qt['s2', 2] = 5.

    return qt


def test_pmf(uniq_qtab, dup_qtab):
    policy = GreedyPolicy(uniq_qtab)
    cases = {
        's0': [0., 1., 0.],
        's1': [1., 0., 0.],
        's2': [0., 0., 1.]
    }
    for state, exp_pmf in cases.items():
        act_pmf = list(policy.pmf(state))
        assert exp_pmf == act_pmf

    # For unknown state, any action can be 1.
    act_pmf = np.array(policy.pmf('unknown'))
    assert 2 == len(act_pmf[act_pmf == 0.])
    assert 1 == len(act_pmf[act_pmf == 1.])

    policy = GreedyPolicy(dup_qtab)
    # For s0, either action 1 or 2 should be 1.0
    act_pmf = policy.pmf('s0')
    assert act_pmf[1] == 1. or act_pmf[2] == 1.

    # For s1, it is as before
    act_pmf = policy.pmf('s1')
    assert [1., 0., 0.] == list(act_pmf)

    # For s2, any action can be 1.
    act_pmf = np.array(policy.pmf('s2'))
    assert 2 == len(act_pmf[act_pmf == 0.])
    assert 1 == len(act_pmf[act_pmf == 1.])


def test_prob(uniq_qtab, dup_qtab):
    policy = GreedyPolicy(uniq_qtab)
    assert 0. == policy.prob('s0', 0)

    assert policy.prob('unknown', 0) in [0., 1.]

    try:
        v = policy.prob('s0', 10)  # NOQA
        assert False
    except IndexError:
        assert True

    try:
        v = policy.prob('unknown', 10)  # NOQA
        assert False
    except IndexError:
        assert True


def test_action(uniq_qtab):
    policy = GreedyPolicy(uniq_qtab)
    assert 1 == policy.action('s0')
