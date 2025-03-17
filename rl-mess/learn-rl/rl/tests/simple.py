from gym.envs.toy_text.discrete import DiscreteEnv


class SimpleEnv(DiscreteEnv):
    """
    State values of this environment are -
    [ 8.03114032 11.17119132  8.92357587 ]
    """

    def __init__(self):
        n_actions = 2
        n_states = 3

        P = [None] * n_states
        for s in range(n_states):
            P[s] = [None] * n_actions
        P[0][0] = [(0.5, 0, 0.0, False), (0.0, 1, 0.0, False), (0.5, 2, 0.0, False)]
        P[0][1] = [(0.0, 0, 0.0, False), (0.0, 1, 0.0, False), (1.0, 2, 0.0, False)]
        P[1][0] = [(0.7, 0, 5, False), (0.1, 1, 0.0, False), (0.2, 2, 0.0, False)]
        P[1][1] = [(0.0, 0, 0.0, False), (0.95, 1, 0.0, False), (0.05, 2, 0.0, False)]
        P[2][0] = [(0.4, 0, 0.0, False), (0.6, 1, 0.0, False), (0.0, 2, 0.0, False)]
        P[2][1] = [(0.3, 0, -1, False), (0.3, 1, 0.0, False), (0.4, 2, 0.0, False)]

        isd = [1.0, 0.0, 0.0]

        super().__init__(n_states, n_actions, P, isd)


def policy(action: int, state: int) -> float:
    # (a, s) => π(a|s)
    probs = {(1, 0): 1.0, (0, 1): 1.0, (0, 2): 1.0}
    return probs.get((action, state), 0.0)
