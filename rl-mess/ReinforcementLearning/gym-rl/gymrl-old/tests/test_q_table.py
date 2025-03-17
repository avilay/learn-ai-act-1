from pytest import fixture
from ..q_table import QTable


@fixture
def qtab():
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


def test_item(qtab):
    # Modify existing state
    qtab['s0', 0] = 42.
    assert qtab['s0', 0] == 42.

    # Set a new state
    qtab['s3', 1] = 21.
    assert qtab['s3', 1] == 21.

    # Try to set value of an invalid action
    try:
        qtab['s0', 5] = 1.
        assert False
    except IndexError:
        assert True

    # Try to get value of an invalid action
    try:
        v = qtab['s0', 3]  # NOQA
        assert False
    except IndexError:
        assert True


def test_states(qtab):
    exp_states = set(['s0', 's1', 's2'])
    act_states = set(qtab.states)
    assert exp_states == act_states


def test_all_values(qtab):
    # Get all values of a known state
    exp_vals = [10., 20., 5.]
    act_vals = list(qtab.all_values('s0'))
    assert exp_vals == act_vals

    # Get all values of an unknown state
    exp_vals = [0., 0., 0.]
    act_vals = list(qtab.all_values('unknown'))
    assert exp_vals == act_vals
