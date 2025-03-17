import functools

from mlmetrics.sqlitemetrics.sqlite_metric import SqliteMetric

from rltk import Kit, RlMetrics
from rltk.algos.planning import evaluate_policy
from rltk.core import Agent
from rltk.gridworld import GridMDP, GridAction, use_gridworld_kit
from rltk.simple import SimpleEnv, SimplePolicy


class FrozenLake(GridMDP):
    def __init__(self, nrows, ncols, terminal_states, goal_states):
        super().__init__(nrows, ncols, terminal_states, goal_states)

    def reward(self, state, action, next_state):
        # reward does not depend on state or action, but only on next_state
        return 1. if self.is_goal(next_state) else 0.

    def trans_prob(self, state, action, next_state):
        if action == GridAction.up() or action == GridAction.down():
            stochastic_actions = [GridAction.left(), GridAction.right()]
        else:
            stochastic_actions = [GridAction.up(), GridAction.down()]
        next_states_probs = {self.next_cell(state, action): 0.8}
        for a in stochastic_actions:
            next_states_probs[self.next_cell(state, a)] = 0.1
        return next_states_probs[next_state] if next_state in next_states_probs else 0.

    @classmethod
    def standard(cls):
        """
        SFFF
        FHFH
        FFFH
        HFFG
        """
        return cls(4, 4,
                   terminal_states=[(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)],
                   goal_states=[(3, 3)])

    @staticmethod
    def optimal_policy():
        return SimplePolicy.deterministic([
            ((0, 0), GridAction.down()),
            ((0, 1), GridAction.right()),
            ((0, 2), GridAction.down()),
            ((0, 3), GridAction.left()),

            ((1, 0), GridAction.down()),
            # (1,1) is a hole
            ((1, 2), GridAction.down()),
            # (1,3) is a hole

            ((2, 0), GridAction.right()),
            ((2, 1), GridAction.down()),
            ((2, 2), GridAction.down()),
            # (2,3) is a hole

            # (3,0) is a hole
            ((3, 1), GridAction.right()),
            ((3, 2), GridAction.right())
            # (3,3) is goal
        ])


def wirecheck(fl):
    for action in fl.actions:
        print(action)
    for state in fl.states:
        print(state)
    print(f'Cell (1,1) is terminal? {fl.is_terminal((1,1))}')
    print(f'Cell (3,3) is goal? {fl.is_goal((3,3))}')

    s = (2, 1)
    a = GridAction.up()
    for s_ in fl.states:
        p = fl.trans_prob(s, a, s_)
        print(f'P(s_={s_} | s={s}, a={a}) = {p}')


def full_example():
    fl = FrozenLake.standard()
    env = SimpleEnv(fl, default_start_state=(0, 0))
    agent = Agent(env)
    policy = FrozenLake.optimal_policy()
    episode = agent.move_to_end(policy)
    episode.calc_cum_rewards(gamma=0.9)
    for t, step in enumerate(episode):
        print(f'{step}, {episode.cum_reward(t):.3f}')


def policy_eval():
    use_gridworld_kit()
    fl = FrozenLake.standard()
    pi = FrozenLake.optimal_policy()
    svals = evaluate_policy(fl, pi)
    print(svals)


def main():
    kit = Kit.instance()
    print(f'Starting with {kit.name}')
    db = '/Users/avilay/temp/metrics/rl.db'
    kit.metric = functools.partial(SqliteMetric, db=db)

    # wirecheck()
    # full_example()
    policy_eval()
    # kit.print_contents()
    # use_gridworld_kit()

    RlMetrics.instance().close()


if __name__ == '__main__':
    main()
