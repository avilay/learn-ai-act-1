import gym
import gym.envs.toy_text.frozen_lake as fl
from rl.common.planning import calc_state_vals
from rl.common.typedefs import Policy


def build_best_policy(env: fl.FrozenLakeEnv) -> Policy:
    s_rc = {}
    for r in range(env.nrow):
        for c in range(env.ncol):
            s = r * env.ncol + c
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

    return opt_policy


def main():
    frozen_lake = gym.make("FrozenLake-v0")
    policy = build_best_policy(frozen_lake)
    svals = calc_state_vals(frozen_lake, policy)
    print(svals)


if __name__ == "__main__":
    main()
