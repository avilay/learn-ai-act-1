from testenvs.simple import SimpleEnv
from testenvs.planning import calc_state_vals
from testenvs.model_free import build_replay_buffer
import gym.envs.toy_text.frozen_lake as fl
import gym


def policy(action: int, state: int) -> float:
    # (a, s) => π(a|s)
    probs = {(1, 0): 1.0, (0, 1): 1.0, (0, 2): 1.0}
    return probs.get((action, state), 0.0)


def test_calc_state_vals():
    env = SimpleEnv()
    svals = calc_state_vals(env, policy)
    print(svals)


def frozen_lake_svals():
    frozen_lake = gym.make("FrozenLake-v0")

    s_rc = {}
    for r in range(frozen_lake.nrow):
        for c in range(frozen_lake.ncol):
            s = r * frozen_lake.ncol + c
            s_rc[s] = (r, c)

    moves = {
        (0, 0): fl.DOWN,
        (0, 1): fl.RIGHT,
        (0, 2): fl.DOWN,
        (0, 3): fl.LEFT,
        (1, 0): fl.DOWN,
        (1, 2): fl.DOWN,
        (2, 0): fl.RIGHT,
        (2, 1): fl.DOWN,
        (2, 2): fl.DOWN,
        (3, 1): fl.RIGHT,
        (3, 2): fl.RIGHT,
    }

    def opt_policy(action, state):
        r, c = s_rc[state]

        if r == 3 and c == 3:
            # in terminal state every action is equally likely
            return 0.25

        if (r, c) not in moves:
            # prob of any action from this state is 0
            return 0.0

        best_move = moves[(r, c)]
        if action == best_move:
            return 1.0
        else:
            return 0.0

    svals = calc_state_vals(frozen_lake, opt_policy)
    for state, sval in enumerate(svals):
        r, c = s_rc[state]
        print(f"({r}, {c}) => {sval:.3f}")

    buf = build_replay_buffer(frozen_lake, opt_policy)
    print(buf[0])


# test_calc_state_vals()
frozen_lake_svals()
