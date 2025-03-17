from time import sleep
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from ..planning import PlanningMetric
from .grid_mdp import GridMDP


class GridVisualizer:
    def __init__(self, grid_mdp: GridMDP):
        self._mdp = grid_mdp

        self._action_keys = {
            'up': 0,
            'down': 1,
            'left': 2,
            'right': 3
        }
        up_arrow = (0, -0.3)
        down_arrow = (0, 0.3)
        left_arrow = (-0.3, 0)
        right_arrow = (0.3, 0)
        self._action_arrows = [up_arrow, down_arrow, left_arrow, right_arrow]

    def _create_grids(self, metric):
        svals_grid = np.zeros((4, 4))
        for state, sval in metric.svals.items():
            cell = self._mdp.state2cell(state)
            svals_grid[cell[0], cell[1]] = sval

        pi_grid = np.zeros((4, 4), np.int)
        for state, action_dist in metric.pidist.items():
            cell = self._mdp.state2cell(state)
            hsh = {v: k for k, v in action_dist.items()}
            action = hsh[1.0]
            pi_grid[cell[0], cell[1]] = self._action_keys[action]

        return svals_grid, pi_grid

    def render(self, context_id, iter_num, svals_grid, norm_svals_grid, pi_grid):
        fig = plt.subplot(111)
        fig.imshow(norm_svals_grid, cmap='RdYlGn')
        for r in range(self._mdp.nrows):
            for c in range(self._mdp.ncols):
                x = c
                y = r

                dx, dy = self._action_arrows[pi_grid[r, c]]
                plt.arrow(x, y, dx, dy, head_width=.1,
                          color='#ED755B', linewidth=0.5)

                state = self._mdp.cell2state((r, c))
                if self._mdp.is_goal(state):
                    label = '*'
                    color = 'y'
                    fontsize = 35
                elif self._mdp.is_terminal(state):
                    label = 'x'
                    color = '#06F1F8'
                    fontsize = 30
                else:
                    label = f'{svals_grid[r,c]:.2f}'
                    color = 'k'
                    fontsize = 8
                fig.text(x, y, label, color=color, horizontalalignment='center',
                         verticalalignment='center', fontsize=fontsize)
        fig.set_xticks(np.arange(self._mdp.nrows) - 0.5)
        fig.set_yticks(np.arange(self._mdp.ncols) - 0.5)
        fig.grid(True, color='c')
        fig.axes.set_xticklabels([])
        fig.axes.set_yticklabels([])
        fig.set_title(f'{context_id:<16} - {iter_num:>3}')
        plt.show()

    def visualize(self, metrics: List[PlanningMetric], sleep_secs=0.1):
        svals_grids = np.zeros(
            (len(metrics), self._mdp.nrows, self._mdp.ncols))
        pi_grids = np.zeros(
            (len(metrics), self._mdp.nrows, self._mdp.ncols), np.int)
        for i, metric in enumerate(metrics):
            svals_grid, pi_grid = self._create_grids(metric)
            svals_grids[i] = svals_grid
            pi_grids[i] = pi_grid

        # Normalize svals to be between 0 and 1
        norm_svals_grids = (svals_grids - np.min(svals_grids)) / np.max(svals_grids)

        for i in range(len(metrics)):
            clear_output(True)
            self.render(metrics[i].context_id, metrics[i].iter_num,
                        svals_grids[i], norm_svals_grids[i], pi_grids[i])
            sleep(sleep_secs)
