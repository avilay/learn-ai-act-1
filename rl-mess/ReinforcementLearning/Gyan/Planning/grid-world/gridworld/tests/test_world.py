import pytest
from pytest import fixture
from ..world import World, State, Action

@fixture
def world():
    return World(3, 3)

def test_states(world):
    expected_states = set()
    for r in range(3):
        for c in range(3):
            expected_states.add(State(row=r, col=c))

    actual_states = set()
    for state in world.states():
        actual_states.add(state)

    assert expected_states == actual_states

def test_actions(world):
    expected_actions = {
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT
    }

    actual_actions = set()
    for action in world.actions():
        actual_actions.add(action)

    assert expected_actions == actual_actions

def test_nums(world):
    assert 9 == world.num_states
    assert 4 == world.num_actions

def test_is_terminal(world):
    assert world.is_terminal(State(row=0, col=0))
    assert world.is_terminal(State(row=2, col=2))
    assert not world.is_terminal(State(row=1, col=1))

def test_terminal_move(world):
    s1 = State(row=0, col=0)
    s2 = State(row=2, col=2)
    for s in [s1, s2]:
        for a in world.actions():
            assert s, 0 == world.move(s, a)

def test_top_edge_move(world):
    top_edge_states = [
        State(row=0, col=1),
        State(row=0, col=2)
    ]
    for s in top_edge_states:
        # Cannot go UP, if agent tries it bounces back in the same state
        assert s, -1 == world.move(s, Action.UP)

        # Can go down
        exp_state = State(row=s.row+1, col=s.col)
        assert exp_state, -1 == world.move(s, Action.DOWN)

def test_bottom_edge_move(world):
    bottom_edge_states = [
        State(row=2, col=0),
        State(row=2, col=1)
    ]
    for s in bottom_edge_states:
        # Can go up
        exp_state = State(row=s.row-1, col=s.col)
        assert exp_state, -1 == world.move(s, Action.UP)

        # Cannot go DOWN, if agent tries it bounces back in the same state
        assert s, -1 == world.move(s, Action.DOWN)

def test_left_edge_move(world):
    left_edge_states = [
        State(row=1, col=0),
        State(row=2, col=0)
    ]
    for s in left_edge_states:
        # Cannot go left
        assert s, -1 == world.move(s, Action.LEFT)

        # Can go right
        exp_state = State(row=s.row, col=s.col+1)
        assert exp_state, -1 == world.move(s, Action.RIGHT)

def test_right_edge_move(world):
    right_edge_states = [
        State(row=0, col=2),
        State(row=1, col=2)
    ]
    for s in right_edge_states:
        # Can go left
        exp_state = State(row=s.row, col=s.col-1)
        assert exp_state, -1 == world.move(s, Action.LEFT)

        # Cannot go right
        assert s, -1 == world.move(s, Action.RIGHT)

def test_non_edge_move(world):
    s = State(row=1, col=1)

    exp_state = State(row=s.row-1, col=s.col)
    assert exp_state, -1 == world.move(s, Action.UP)

    exp_state = State(row=s.row+1, col=s.col)
    assert exp_state, -1 == world.move(s, Action.DOWN)

    exp_state = State(row=s.row, col=s.col-1)
    assert exp_state, -1 == world.move(s, Action.LEFT)

    exp_state = State(row=s.row, col=s.col+1)
    assert exp_state, -1 == world.move(s, Action.RIGHT)

def test_visualize(world):
    world.visualize()
    assert True

    values = {
        State(row=0, col=0): 0.,
        State(row=0, col=1): 1.234567,
        State(row=0, col=2): 2.0987656,
        State(row=1, col=0): 3.87583058,
        State(row=1, col=1): 4.18289347,
        State(row=1, col=2): 5.901938445,
        State(row=2, col=0): 6.128347849,
        State(row=2, col=1): 7.1823,
        State(row=2, col=2): 0.
    }
    fmt_vals = {k: f'{v:.3f}' for k, v in values.items()}
    world.visualize(fmt_vals)
