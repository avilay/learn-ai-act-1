import pytest
from rltk.simple import SimpleMDP
import itertools
from rltk.core import MdpFunctionError


@pytest.fixture
def probs(states, actions):
    s0, s1, s2 = states
    a0, a1 = actions
    return {
        (s0, a0): {s1: 0.8, s2: 0.2},
        (s0, a1): {s0: 0.6, s2: 0.4},
        (s1, a0): {s1: 0.15, s2: 0.85},
        (s1, a1): {s2: 1.0}
    }


@pytest.fixture
def reward():
    return lambda s, a, s_: 5 if s_ == 's2' else -1


def test_ctor(states, actions, probs, reward):
    s0, s1, s2 = states
    a0, a1 = actions
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward
    )
    assert mdp

    # Bad terminal state
    with pytest.raises(ValueError):
        mdp = SimpleMDP(
            states,
            actions,
            terminal_states=['snew'],
            probs=probs,
            reward=reward
        )

    # Probs don't sum to 1
    with pytest.raises(ValueError):
        probs = {
            (s0, a0): {s1: 0.8, s2: 0.5},
            (s0, a1): {s0: 0.6, s2: 0.4},
            (s1, a0): {s1: 0.15, s2: 0.85},
            (s1, a1): {s2: 1.0}
        }
        mdp = SimpleMDP(
            states,
            actions,
            terminal_states=[s2],
            probs=probs,
            reward=reward
        )


def test_reward(states, actions, probs, reward):
    s0, s1, s2 = states
    a0, a1 = actions
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward
    )
    for s, a, s_ in itertools.product([s0, s1], [a0, a1], [s0, s1, s2]):
        exp_reward = 5 if s_ == s2 else -1
        assert exp_reward == mdp.reward(s, a, s_)

    with pytest.raises(ValueError):
        mdp.reward(s2, a0, s1)

    def reward_func(s, a, s_): raise RuntimeError('KABOOM!')
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward_func
    )
    with pytest.raises(MdpFunctionError):
        mdp.reward(s0, a1, s2)


def test_prob(states, actions, probs, reward):
    s0, s1, s2 = states
    a0, a1 = actions
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward
    )
    assert 0.6 == mdp.trans_prob(s0, a1, s0)
    assert 0.0 == mdp.trans_prob(s0, a1, s1)
    with pytest.raises(ValueError):
        mdp.trans_prob(s2, a0, s1)


def test_state_action(states, actions, probs, reward):
    s0, s1, s2 = states
    a0, a1 = actions
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward
    )
    states = [s0, s1, s2]
    for s in mdp.states:
        assert s in states
        states.remove(s)

    actions = [a0, a1]
    for a in mdp.actions:
        assert a in actions
        actions.remove(a)
