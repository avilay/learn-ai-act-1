from rltk.core.discrete_mdp import DiscreteMDP
from rltk.core.types import State, Action, Reward
from typing import Set, Sequence, Tuple, Dict
from abc import abstractmethod
from copy import copy

Cell = Tuple[int, int]


class GridMDP(DiscreteMDP):
    """
    Abstract base class to define MDPs for grid worlds. Concrete classes
    must implement reward and dist methods.
    """
    def __init__(self, nrows, ncols) -> None:
        self._actions = {'up', 'down', 'left', 'right'}

        self._nrows = nrows
        self._ncols = ncols
        self._states: Dict[str, Cell] = {}
        for r in range(self._nrows):
            for c in range(self._ncols):
                cell = (r, c)
                self._states[self.cell2state(cell)] = cell

        self._goals: Set[Cell] = set()
        self._terminals: Set[Cell] = set()

    @property
    def nrows(self) -> int:
        return self._nrows

    @property
    def ncols(self) -> int:
        return self._ncols

    @property
    def goals(self) -> Set[Cell]:
        return self._goals

    @goals.setter
    def goals(self, val: Set[Cell]) -> None:
        self._goals = copy(val)

    @property
    def terminals(self) -> Set[Cell]:
        return self._terminals

    @terminals.setter
    def terminals(self, val: Set[Cell]) -> None:
        self._terminals = copy(val)

    @property
    def states(self) -> Set[State]:
        return set(self._states.keys())

    @property
    def actions(self) -> Set[State]:
        return self._actions

    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> Reward:
        pass

    @abstractmethod
    def dist(self, state: State, action: Action) -> Sequence[Tuple[State, float]]:
        pass

    def is_terminal(self, state: State):
        return self._states[state] in self._terminals

    def is_goal(self, state: State):
        return self._states[state] in self.goals

    def state2cell(self, state: State) -> Cell:
        return self._states[state]

    def cell2state(self, state: Cell) -> State:
        return f'({state[0]},{state[1]})'

    def next_cell(self, cell: Cell, action) -> Cell:
        r, c = cell
        if action == 'up':
            r = max(0, r-1)
        elif action == 'down':
            r = min(r+1, self._nrows-1)
        elif action == 'left':
            c = max(0, c-1)
        elif action == 'right':
            c = min(c+1, self._ncols-1)
        else:
            raise RuntimeError("KABOOM!")
        return (r, c)
