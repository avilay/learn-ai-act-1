from rltk.core import Agent, Step, State, Action
from rltk.simple import SimpleEnv, SimplePolicy, SimpleMDP
import pytest

State.register(str)
Action.register(str)


@pytest.fixture
def states():
    return ['s0', 's1', 's2']


@pytest.fixture
def actions():
    return ['a0', 'a1']


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


@pytest.fixture
def policy(states, actions):
    s0, s1, _ = states
    a0, a1 = actions
    return SimplePolicy({
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a0, 0.1),
        (s1, a1, 0.9)
    })


def test_move(states, actions, probs, reward, policy):
    s0, s1, s2 = states
    a0, a1 = actions
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward
    )
    env = SimpleEnv(mdp, s0)
    agent = Agent(env)
    step = agent.move(policy)
    exp_steps = [
        Step(state='s0', action='a0', reward=-1, next_state='s1'),
        Step(state='s0', action='a0', reward=5, next_state='s2'),
        Step(state='s0', action='a1', reward=-1, next_state='s0'),
        Step(state='s0', action='a1', reward=5, next_state='s2'),
    ]
    assert step in exp_steps, f'Actual step was {step}'


def test_move_to_end(states, actions, probs, reward, policy):
    s0, s1, s2 = states
    a0, a1 = actions
    mdp = SimpleMDP(
        states,
        actions,
        terminal_states=[s2],
        probs=probs,
        reward=reward
    )
    env = SimpleEnv(mdp, s0)
    agent = Agent(env)
    episode = agent.move_to_end(policy)
    steps = list(episode)
    for step in steps[:-1]:
        assert step.next_state != 's2'
        assert step.reward == -1
    final_step = steps[-1]
    assert final_step.next_state == 's2'
    assert final_step.reward == 5

    episode.calc_cum_rewards()
    num_steps = len(episode)
    for t in range(num_steps):
        r = episode.cum_reward(t)
        assert r <= 5
