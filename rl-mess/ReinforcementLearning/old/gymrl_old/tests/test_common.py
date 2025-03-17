import numpy as np
import gym
import gym.spaces
from ..common import Step, calc_cum_rewards, gen_greedy_policy, gen_episode, gen_epsilon_greedy_policy


np.random.seed(1)


def test_calc_cum_rewards():
    episode = [
        Step(state='s0', action=0),
        Step(reward=10., state='s1', action=1),
        Step(reward=20., state='s2', action=2),
        Step(reward=30., state='s3', action=3)
    ]

    exp_G = np.array([60., 50., 30., 0.])
    act_G = calc_cum_rewards(episode)
    assert np.array_equal(exp_G, act_G)

    exp_G = np.array([12.3, 23., 30., 0.])
    act_G = calc_cum_rewards(episode, γ=0.1)
    assert np.array_equal(exp_G, act_G)


def test_gen_greedy_policy():
    Q = {
        's0': np.array([10., 20., 5.]),
        's1': np.array([15., 3., 5.]),
        's2': np.array([1., 2., 5.])
    }
    exp_π = {'s0': 1, 's1': 0, 's2': 2}
    act_π = gen_greedy_policy(Q)
    assert exp_π == act_π

    Q = {
        's0': np.array([10., 20., 20.]),
        's1': np.array([15., 3., 5.]),
        's2': np.array([5., 5., 5.])
    }
    exp_π = {'s0': 2, 's1': 0, 's2': 0}
    act_π = gen_greedy_policy(Q)
    assert exp_π == act_π


def test_gen_episode():
    """
    [00] [01] [02] [03]
     S    F    F    F

    [04] [05] [06] [07]
     F    H    F    H

    [08] [09] [10] [11]
     F    F    F    H

    [12] [13] [14] [15]
     H    F    F    G
    """
    frozen_lake = gym.make('FrozenLake-v0')
    Step.action_labels = ['LEFT', 'DOWN', 'RIGHT', 'UP']
    LEFT = 0  # NOQA
    DOWN = 1
    RIGHT = 2
    UP = 3  # NOQA
    moves = [
        DOWN,   # In 00 go DOWN
        RIGHT,  # In 01 go RIGHT
        DOWN,   # In 02 go DOWN
        LEFT,   # In 03 go LEFT
        DOWN,   # In 04 go DOWN
        None,   # 05 is a hole
        DOWN,   # In 06 go DOWN
        None,   # 07 is a hole
        RIGHT,  # In 08 go RIGHT
        DOWN,   # In 09 go DOWN
        DOWN,   # In 10 go DOWN
        None,   # 11 is a hole
        None,   # 12 is a hole
        RIGHT,  # In 13 go RIGHT
        RIGHT,  # In 14 go RIGHT
        None    # 15 is the goal
    ]

    def policy(state, action=None):
        pmf = np.zeros(4)
        pmf[moves[state]] = 1.0
        return pmf

    episode = gen_episode(frozen_lake, policy)
    for step in episode:
        exp_action = moves[step.state]
        act_action = step.action
        assert exp_action == act_action


def test_gen_epsilon_greedy_policy():
    Q = {
        's0': np.array([10., 20., 5.]),
        's1': np.array([15., 3., 5.]),
        's2': np.array([1., 2., 5.])
    }
    policy = gen_epsilon_greedy_policy(Q, ε=0.1)
    state_action_probs = {
        's0': np.array([0.033, 0.933, 0.033]),
        's1': np.array([0.933, 0.033, 0.033]),
        's2': np.array([0.033, 0.033, 0.933])
    }
    for state in ['s0', 's1', 's2']:
        for action in range(3):
            exp_prob = state_action_probs[state][action]
            act_prob = policy(state, action)
            np.testing.assert_almost_equal(exp_prob, act_prob, decimal=3)
    pmf = policy('s10')
    print(pmf)

    Q = {
        's0': np.array([10., 20., 20.]),
        's1': np.array([15., 3., 5.]),
        's2': np.array([5., 5., 5.])
    }
    policy = gen_epsilon_greedy_policy(Q, ε=0.1)
    state_action_probs = {
        's0': np.array([0.033, 0.033, 0.933]),
        's1': np.array([0.933, 0.033, 0.033]),
        's2': np.array([0.933, 0.033, 0.033]),
    }
    for state in ['s0', 's1', 's2']:
        exp_probs = state_action_probs[state]
        act_probs = policy(state)
        np.testing.assert_almost_equal(exp_probs, act_probs, decimal=3)
