import gym
import pytest

from utils import StepBuffer


def test_ctor():
    env = gym.make("FrozenLake-v0")
    buf = StepBuffer(env)
    assert True
    with pytest.raises(ValueError):
        StepBuffer(env, capacity=-1)
    with pytest.raises(ValueError):
        StepBuffer(env, avg_over=-1)
    with pytest.raises(ValueError):
        StepBuffer(env, capacity=-1, avg_over=-1)


def test_replinish():
    env = gym.make("FrozenLake-v0")
    buf = StepBuffer(env, capacity=10)
    policy = lambda s: env.action_space.sample()
    buf.replinish(policy, 0.5)
    assert len(buf._buffer) == 5


def test_sample():
    env = gym.make("FrozenLake-v0")
    buf = StepBuffer(env)
    policy = lambda s: env.action_space.sample()
    buf.replinish(policy, 0.5)
    batch = buf.sample(100)
    assert len(batch) == 100

    buf = StepBuffer(env, capacity=10)
    with pytest.raises(ValueError):
        buf.sample(100)


def test_scores():
    env = gym.make("CartPole-v0")
    buf = StepBuffer(env)
    policy = lambda s: env.action_space.sample()
    buf.replinish(policy, 0.5)
    assert buf.avg_score
    assert buf.latest_score
