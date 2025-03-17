import math
import random
from collections import defaultdict
from typing import Dict, Tuple, Iterator

from rltk.core import State, StateValues


class SimpleStateValues(StateValues):
    def __init__(self) -> None:
        self._svals: Dict[State, float] = defaultdict(float)

    def __getitem__(self, state: State) -> float:
        return self._svals[state]

    def __setitem__(self, state: State, value: float) -> None:
        self._svals[state] = value

    def __len__(self) -> int:
        return len(self._svals)

    def __add__(self, value: float) -> StateValues:
        svals: Dict[State, float] = defaultdict(float)
        for s, v in self._svals.items():
            svals[s] = v + value
        obj: SimpleStateValues = SimpleStateValues()
        obj._svals = svals
        return obj

    def __iter__(self) -> Iterator[Tuple[State, float]]:
        for state in self._svals.keys():
            yield state, self._svals[state]

    def __str__(self):
        ret = 'State Values:\n'
        for state, value in self._svals.items():
            ret += f'\t{state} => {value:.3f}\n'
        return ret

    def are_close(self, other: StateValues) -> bool:
        if (len(self) == 0 and len(other) == 0) or len(self) != len(other):
            return False
        for s in self._svals.keys():
            v1 = self._svals[s]
            v2 = other[s]
            if not math.isclose(v1, v2, rel_tol=0.001, abs_tol=0.001):
                return False
        return True
