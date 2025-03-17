from rltk.core import Episode, Step
import pytest


@pytest.fixture
def steps(states, actions):
    s0, s1, s2 = states
    a0, a1 = actions

    step0 = Step(
        state=s0,
        action=a0,
        reward=-1,
        next_state=s1
    )
    step1 = Step(
        state=s1,
        action=a0,
        reward=-1,
        next_state=s1
    )
    step2 = Step(
        state=s1,
        action=a1,
        reward=5,
        next_state=s2
    )
    return [step0, step1, step2]


@pytest.fixture
def episode(steps):
    ep = Episode()
    for step in steps:
        ep.append(step)
    return ep


def test_append_iter_len(steps):
    ep = Episode()

    assert 0 == len(ep)
    for step in ep:
        assert False

    for step in steps:
        ep.append(step)
    assert len(steps) == len(ep)
    for i, step in enumerate(ep):
        assert steps[i] == step

    with pytest.raises(ValueError):
        ep.append(None)


def test_calc_cum_rewards(steps, episode):
    with pytest.raises(RuntimeError):
        episode.cum_reward(0)

    episode.calc_cum_rewards()
    assert 3 == episode.cum_reward(0)
    assert 4 == episode.cum_reward(1)
    assert 5 == episode.cum_reward(2)

    episode.append(steps[-1])
    with pytest.raises(RuntimeError):
        episode.cum_reward(0)
