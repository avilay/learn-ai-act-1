from collections import namedtuple, defaultdict
from enum import Enum, auto
from typing import Any, Iterator, Mapping, Tuple, Dict

State = namedtuple('State', ['row', 'col'])

class Action(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class World:
    def __init__(self, nrows: int, ncols: int) -> None:
        self._nrows = nrows
        self._ncols = ncols

    def what_if_move(self, s: State, a: Action) -> Tuple[Mapping[State, float], float]:
        state_dist: Dict[State, float] = {}
        for state in self.states():
            state_dist[state] = 0.
        s_, r = self.move(s, a)
        state_dist[s_] = 1.
        return state_dist, r

    def move(self, s: State, a: Action) -> Tuple[State, float]:
        # The state transition probability is deterministic
        if self.is_terminal(s):
            return s, 0

        if a == Action.UP:
            new_row = s.row - 1 if s.row > 0 else s.row
            new_col = s.col
        elif a == Action.LEFT:
            new_row = s.row
            new_col = s.col - 1 if s.col > 0 else s.col
        elif a == Action.DOWN:
            new_row = s.row + 1 if s.row < (self._nrows - 1) else s.row
            new_col = s.col
        elif a == Action.RIGHT:
            new_row = s.row
            new_col = s.col + 1 if s.col < (self._ncols - 1) else s.col
        return State(row=new_row, col=new_col), -1

    def states(self) -> Iterator[State]:
        for r in range(self._nrows):
            for c in range(self._ncols):
                yield State(row=r, col=c)

    def actions(self) -> Iterator[Action]:
        for a in Action:
            yield a

    @property
    def num_states(self) -> int:
        return self._ncols * self._nrows

    @property
    def num_actions(self) -> int:
        return len(Action)

    def is_terminal(self, s: State) -> bool:
        return (s.row == 0 and s.col == 0) or (s.row == self._nrows-1 and s.col == self._ncols-1)

    def visualize(self, cells=defaultdict(lambda: 'o')) -> None:
        print()
        for c in range(self._ncols):
            for r in range(self._nrows):
                print(cells[State(row=r, col=c)], end='\t')
            print()

