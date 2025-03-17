import math
from pprint import pprint
from copy import copy
from collections import defaultdict
from toymdp import MDP, Policy, Simulator, value_iteration, evaluate_policy

MAX_ROW = 3
MAX_COL = 3

def rewards(state, action, next_state):
        if next_state == (3,3):
            return 1
        else:
            return 0


def next_cell(row, col, action):
    r, c = row, col
    if action == 'up':
        r = max(0, r-1)
    elif action == 'down':
        r = min(r+1, MAX_ROW)
    elif action == 'left':
        c = max(0, c-1)
    elif action == 'right':
        c = min(c+1, MAX_COL)
    else:
        raise RuntimeError("KABOOM!")
    return r, c


def probs(states, actions):
    hsh = defaultdict(list)

    for state in states:
        for action in actions:
            stochastic_actions = []
            if action == 'up' or action == 'down':
                stochastic_actions = ['left', 'right']
            elif action == 'left' or action == 'right':
                stochastic_actions = ['up', 'down']

            hsh[(state,action)].append((next_cell(*state, action), 0.8))
            for stochastic_action in stochastic_actions:
                r, c = next_cell(*state, stochastic_action)
                hsh[(state,action)].append(((r,c), 0.1))

    return hsh


def main():
    """
    SFFF
    FHFH
    FFFH
    HFFG
    """
    states = {(0,0), (0,1), (0,2), (0,3),
              (1,0), (1,1), (1,2), (1,3),
              (2,0), (2,1), (2,2), (2,3),
              (3,0), (3,1), (3,2), (3,3)}
    actions = {'up', 'down', 'left', 'right'}
    terminal_states = {(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)}
    mdp = MDP(states, actions, rewards, probs(states, actions), terminal_states)
    state_vals = value_iteration(mdp)
    pprint(state_vals)

    rules = {
        (0,0): [('down', 1.)],
        (0,1): [('right', 1.)],
        (0,2): [('down', 1.)],
        (0,3): [('left', 1.)],

        (1,0): [('down', 1.)],
        # (1,1) is a hole
        (1,2): [('down', 1.)],
        # (1,3) is a hole

        (2,0): [('right', 1.)],
        (2,1): [('down', 1.)],
        (2,2): [('down', 1.)],
        # (2,3) is a hole

        # (3,0) is a hole
        (3,1): [('right', 1.0)],
        (3,2): [('right', 1.0)],
        # (3,3) is goal
    }

    policy = Policy(rules)
    state_vals = evaluate_policy(mdp, policy)
    pprint(state_vals)


if __name__ == '__main__':
    main()
