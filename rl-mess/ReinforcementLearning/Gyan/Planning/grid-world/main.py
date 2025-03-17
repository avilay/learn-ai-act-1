import logging
from gridworld import World, State, Action, Policy, evaluate_policy, gen_episode

logging.basicConfig(level=logging.DEBUG)

def main():
    world = World(nrows=4, ncols=4)
    def pmf(s):
        return {
            Action.UP: 0.25,
            Action.DOWN: 0.25,
            Action.LEFT: 0.25,
            Action.RIGHT: 0.25
        }
    policy = Policy(pmf)
    values = evaluate_policy(policy, world)
    fmt_vals = {k: f'{v:.4f}' for k, v in values.items()}
    world.visualize(fmt_vals)

if __name__ == '__main__':
    main()
