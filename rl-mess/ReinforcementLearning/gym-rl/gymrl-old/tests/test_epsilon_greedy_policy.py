from pytest import fixture
import math
from ..epsilon_greedy_policy import EpsilonGreedyPolicy
from ..q_table import QTable
import numpy as np


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
    policy = EpsilonGreedyPolicy(uniq_qtab, ε=0.1)
    cases = {
        's0': np.array([0.033, 0.933, 0.033]),
        's1': np.array([0.933, 0.033, 0.033]),
        's2': np.array([0.033, 0.033, 0.933])
    }
    for state, exp_pmf in cases.items():
        act_pmf = np.array(policy.pmf(state))
        np.testing.assert_almost_equal(exp_pmf, act_pmf, decimal=3)

    # For unknown state, any action can be 0.933
    act_pmf = np.array(policy.pmf('unknown'))
    assert 1 == len(act_pmf[act_pmf >= 0.933])
    assert 2 == len(act_pmf[act_pmf <= 0.034])

    policy = EpsilonGreedyPolicy(dup_qtab, ε=0.1)
    # For s0 either action 1 or 2 should be 0.933
    act_pmf = policy.pmf('s0')
    assert math.isclose(act_pmf[1], 0.933, abs_tol=3, rel_tol=3) or math.isclose(act_pmf[2], 0.933, abs_tol=3, rel_tol=3)

    # For s1 it is as before
    act_pmf = policy.pmf('s1')
    np.testing.assert_almost_equal(np.array([0.933, 0.033, 0.033]), act_pmf, decimal=3)

    # For s2, any action can be 0.933
    act_pmf = np.array(policy.pmf('s2'))
    assert 1 == len(act_pmf[act_pmf >= 0.933])
    assert 2 == len(act_pmf[act_pmf <= 0.034])
