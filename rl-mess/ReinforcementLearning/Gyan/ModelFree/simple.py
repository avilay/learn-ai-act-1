"""
transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's1': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}
"""
from pprint import pprint
from toymdp import MDP, value_iteration, Policy, evaluate_policy, gen_greedy_policy


def main():
    states = {'s0', 's1', 's2'}
    actions = {'a0', 'a1'}
    rhsh = {
        's1': {'a0': {'s0': +5}},
        's2': {'a1': {'s0': -1}}
    }
    rewards = lambda s, a, s_: rhsh.get(s, {}).get(a, {}).get(s_, 0.)
    probs = {
        ('s0', 'a0'): [('s0', 0.5), ('s2', 0.5)],
        ('s0', 'a1'): [('s2', 1.0)],
        ('s1', 'a0'): [('s0', 0.7), ('s1', 0.1), ('s2', 0.2)],
        ('s1', 'a1'): [('s1', 0.95), ('s2', 0.05)],
        ('s2', 'a0'): [('s0', 0.4), ('s1', 0.6)],
        ('s2', 'a1'): [('s0', 0.3), ('s1', 0.3), ('s2', 0.4)],
    }
    mdp = MDP(states, actions, rewards, probs, {})
    optimal_svals = value_iteration(mdp)
    pprint(optimal_svals)

    optimal_policy = gen_greedy_policy(mdp, optimal_svals)
    # for s in mdp.states:
    #     pprint(s)
    #     pprint(optimal_policy.dist(s))

    state_vals = evaluate_policy(mdp, optimal_policy)
    # pprint(state_vals)


if __name__ == '__main__':
    main()
