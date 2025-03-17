from abc import abstractmethod
from rltk.core import DiscreteMDP, State, Action
from typing import Iterator, Tuple, Sequence
import itertools
from .grid_action import GridAction

Cell = Tuple[int, int]
State.register(tuple)


class GridMDP(DiscreteMDP):
    def __init__(
        self,
        nrows: int,
        ncols: int,
        terminal_states: Sequence[Cell],
        goal_states: Sequence[Cell]
    ) -> None:
        self._nrows = nrows
        self._ncols = ncols

        for i, j in terminal_states:
            if i < 0 or i > nrows or j < 0 or j > ncols:
                raise ValueError(f'Invalid terminal state {i}{j}')
        self._terminal_states = set(terminal_states)
        self._goal_states = set(goal_states)
        if not self._goal_states.issubset(self._terminal_states):
            raise ValueError('Goals must also be terminal!')

    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> float:
        raise NotImplementedError()

    @abstractmethod
    def trans_prob(self, state: State, action: Action, next_state: State) -> float:
        raise NotImplementedError()

    @property
    def states(self) -> Iterator[State]:
        for i, j in itertools.product(range(self._nrows), range(self._ncols)):
            yield i, j

    @property
    def actions(self) -> Iterator[Action]:
        yield GridAction.up()
        yield GridAction.down()
        yield GridAction.left()
        yield GridAction.right()

    @property
    def num_rows(self) -> int:
        return self._nrows

    @property
    def num_cols(self) -> int:
        return self._ncols

    def is_terminal(self, state: State) -> bool:
        return state in self._terminal_states

    def is_goal(self, state: State) -> bool:
        return state in self._goal_states

    def next_cell(self, cell: Cell, action: Action) -> Cell:
        r, c = cell
        if action == GridAction.up():
            r = max(0, r-1)
        elif action == GridAction.down():
            r = min(r+1, self._nrows-1)
        elif action == GridAction.left():
            c = max(0, c-1)
        elif action == GridAction.right():
            c = min(c+1, self._ncols-1)
        else:
            raise RuntimeError(f'Unable to take action {action} from state ({r},{c}) ')
        return r, c
