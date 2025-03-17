from rltk.core import Agent, Step
import pytest
from rltk.core import Policy, DiscreteMDP, Env
import numpy as np


class UTPolicy(Policy):
    def __init__(self):
        self._dist = {
            's0': {
                'a0': 0.6,
                'a1': 0.4
            },
            's1': {
                'a0': 0.1,
                'a1': 0.9
            }
        }

    def prob(self, action, given):
        return self._dist[given][action]

    def action(self, state):
        actions = []
        probs = []
        for action, prob in self._dist[state].items():
            actions.append(action)
            probs.append(prob)
        return np.random.choice(actions, p=probs)

    @classmethod
    def deterministic(cls, rules):
        raise NotImplementedError()


class UTMDP(DiscreteMDP):
    def __init__(self, probs, terminal_states):
        self.probs = probs
        self.terminal_states = terminal_states

    def reward(self, s, a, s_):
        return 5 if s_ == 's2' else -1

    def trans_prob(self, s, a, s_):
        return self.probs.get((s, a), {}).get(s_, 0.)

    def is_terminal(self, s):
        return s in self.terminal_states

    @property
    def states(self):
        return ['s0', 's1', 's2']

    @property
    def actions(self):
        return ['a0', 'a1']


class UTEnv(Env):
    def __init__(self, mdp):
        self._mdp = mdp
        self._positions = {}  # key is agent_id and value is the current state

    def enter(self, agent_id):
        self._positions[agent_id] = 's0'

    def curr_state(self, agent_id):
        return self._positions[agent_id]

    def move(self, agent_id, action):
        s = self.curr_state(agent_id)
        states = ['s0', 's1', 's2']
        probs = []
        for state in states:
            probs.append(self._mdp.trans_prob(s, action, state))
        s_ = np.random.choice(states, p=probs)
        r = self._mdp.reward(s, action, s_)
        return r, s_

    def is_terminal(self, state):
        return self._mdp.is_terminal(state)


@pytest.fixture
def policy():
    return UTPolicy()


@pytest.fixture
def env():
    probs = {
        ('s0', 'a0'): {'s0': 0.0, 's1': 0.8, 's2': 0.2},
        ('s0', 'a1'): {'s0': 0.6, 's1': 0.0, 's2': 0.4},
        ('s1', 'a0'): {'s0': 0.0, 's1': 0.15, 's2': 0.85},
        ('s1', 'a1'): {'s0': 0.0, 's1': 0.0, 's2': 1.0}
    }
    mdp = UTMDP(probs, terminal_states=['s2'])
    return UTEnv(mdp)


def test_move(env, policy):
    """
    agent will start in state s0
    Based on the policy it will choose action a0 with prob 0.6
    and action a1 with prob 0.4
    If it chooses a0, then based on the MDP it can land in s1 with prob 0.8
    and s2 with prob 0.2
    If it chooses a1, then it can land in s0 with prob 0.6 and s2 with prob 0.4
    If it lands in s2 then it gets a reward of 5 otherwise -1.
    So the possible steps are -
    s=s0, a=a0, s_=s1, r=-1
                s_=s2, r=5
          a=a1, s_=s0, r=-1
                s_=s2, r=5
    """
    agent = Agent(env)
    step = agent.move(policy)
    exp_steps = [
        Step(state='s0', action='a0', reward=-1, next_state='s1'),
        Step(state='s0', action='a0', reward=5, next_state='s2'),
        Step(state='s0', action='a1', reward=-1, next_state='s0'),
        Step(state='s0', action='a1', reward=5, next_state='s2'),
    ]
    assert step in exp_steps, f'Actual step was {step}'


def test_move_to_end(env, policy):
    agent = Agent(env)
    episode = agent.move_to_end(policy)
    steps = list(episode)
    for step in steps[:-1]:
        assert step.next_state != 's2'
        assert step.reward == -1
    final_step = steps[-1]
    assert final_step.next_state == 's2'
    assert final_step.reward == 5
