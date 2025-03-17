from pprint import pprint
from typing import List, Sequence, Tuple

import rltk.metrics as metrics
from rltk.core import Policy
from rltk.core.types import Action, Reward, State
from rltk.gridworld.grid_mdp import Cell, GridMDP
from rltk.planning import Planning, SqlitePlanningMetricWriter


class FrozenLake(GridMDP):
    def __init__(self, nrows, ncols):
        super().__init__(nrows, ncols)

    def reward(self, state: State, action: Action, next_state: State) -> Reward:
        # reward does not depend on state or action, but only on next_state
        return 1. if self.is_goal(next_state) else 0.

    def dist(self, state: State, action: Action) -> Sequence[Tuple[State, float]]:
        if action == 'up' or action == 'down':
            stochastic_actions = ['left', 'right']
        else:
            stochastic_actions = ['up', 'down']

        s: Cell = self.state2cell(state)
        probs: List[Tuple[State, float]] = []
        s_: Cell = self.next_cell(s, action)
        probs.append((self.cell2state(s_), 0.8))
        for a in stochastic_actions:
            s_ = self.next_cell(s, a)
            probs.append((self.cell2state(s_), 0.1))
        return probs


def main():
    """
    SFFF
    FHFH
    FFFH
    HFFG
    """
    # metrics.set_writer(metrics.ConsoleWriter())
    writer = SqlitePlanningMetricWriter('/Users/avilay/temp/rltk.db')
    metrics.set_writer(writer)

    fl = FrozenLake(4, 4)
    fl.goals = {(3,3)}
    fl.terminals = {(1,1), (1,3), (2,3), (3,0), (3,3)}
    planning = Planning(fl)
    # svals = planning.value_iteration()
    svals = planning.policy_iteration()
    pprint(svals)

    print(f'Metrics saved under run id: {writer.run_id}')
    writer.close()


def main2():
    writer = SqlitePlanningMetricWriter('/Users/avilay/temp/rltk.db')
    metrics.set_writer(writer)

    fl = FrozenLake(4, 4)
    fl.goals = {(3, 3)}
    fl.terminals = {(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)}
    planning = Planning(fl)

    rules = {
        fl.cell2state((0, 0)): [('down', 1.)],
        fl.cell2state((0, 1)): [('right', 1.)],
        fl.cell2state((0, 2)): [('down', 1.)],
        fl.cell2state((0, 3)): [('left', 1.)],

        fl.cell2state((1, 0)): [('down', 1.)],
        # (1,1) is a hole
        fl.cell2state((1, 2)): [('down', 1.)],
        # (1,3) is a hole

        fl.cell2state((2, 0)): [('right', 1.)],
        fl.cell2state((2, 1)): [('down', 1.)],
        fl.cell2state((2, 2)): [('down', 1.)],
        # (2,3) is a hole

        # (3,0) is a hole
        fl.cell2state((3, 1)): [('right', 1.0)],
        fl.cell2state((3, 2)): [('right', 1.0)],
        # (3,3) is goal
    }
    policy = Policy(rules)
    svals = planning.evaluate_policy(policy)
    pprint(svals)

    print(f'Metrics saved under run id: {writer.run_id}')
    writer.close()


if __name__ == '__main__':
    main2()
