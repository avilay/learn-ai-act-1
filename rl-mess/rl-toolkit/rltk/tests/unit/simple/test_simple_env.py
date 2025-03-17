import pytest
from rltk.simple import SimpleEnv
from rltk.core import DuplicateAgentError, DiscreteMDP, UnknownAgentError
import uuid


@pytest.fixture
def env():
    class TestMDP(DiscreteMDP):
        def __init__(self):
            self.probs = {
                ('s0', 'a0'): {'s1': 0.8, 's2': 0.2},
                ('s0', 'a1'): {'s0': 0.6, 's2': 0.4},
                ('s1', 'a0'): {'s1': 0.15, 's2': 0.85},
                ('s1', 'a1'): {'s2': 1.0}
            }

        def reward(self, s, a, s_):
            return 5 if s_ == 's2' else -1

        def trans_prob(self, s, a, s_):
            return self.probs.get((s, a), {}).get(s_, 0.)

        def is_terminal(self, s):
            return s == 's2'

        @property
        def states(self):
            return ['s0', 's1', 's2']

        @property
        def actions(self):
            return ['a0', 'a1']

    return SimpleEnv(TestMDP(), 's0')


def test_enter(env):
    agent_id = uuid.uuid4()
    assert 's0' == env.enter(agent_id)
    with pytest.raises(DuplicateAgentError):
        env.enter(agent_id)


def test_curr_state(env):
    agent_id = uuid.uuid4()
    env.enter(agent_id)
    assert 's0' == env.curr_state(agent_id)
    with pytest.raises(UnknownAgentError):
        env.curr_state(uuid.uuid4())


def test_is_terminal(env):
    assert env.is_terminal('s2')
    assert not env.is_terminal('s0')


def test_move(states, actions, env):
    """
            ('s0', 'a0'): {'s1': 0.8, 's2': 0.2},
            ('s0', 'a1'): {'s0': 0.6, 's2': 0.4},
            ('s1', 'a0'): {'s1': 0.15, 's2': 0.85},
            ('s1', 'a1'): {'s2': 1.0}
    """
    agent_id = uuid.uuid4()
    env.enter(agent_id)
    r, s_ = env.move(agent_id, 'a0')
    assert s_ in ['s1', 's2']
    if s_ == 's2':
        assert 5 == r
    else:
        assert -1 == r
