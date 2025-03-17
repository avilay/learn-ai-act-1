import math
import random
from collections import defaultdict
from typing import Dict, Tuple, Iterator

from rltk.core import ActionValues, State, Action


class SimpleActionValues(ActionValues):
    def __init__(self) -> None:
        self._qvals: Dict[State, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        self._len = 0

    def __getitem__(self, key: Tuple[State, Action]) -> float:
        s, a = key
        return self._qvals[s][a]

    def __setitem__(self, key: Tuple[State, Action], value: float) -> None:
        s, a = key
        self._qvals[s][a] = float(value)
        self._len += 1

    def __len__(self) -> int:
        return self._len

    def __add__(self, value: float) -> ActionValues:
        qvals: Dict[State, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        for s, avals in self._qvals.items():
            for a, v in avals.items():
                qvals[s][a] = v + value
        obj: SimpleActionValues = SimpleActionValues()
        obj._qvals = qvals
        return obj

    def __iter__(self) -> Iterator[Tuple[State, Action, float]]:
        for state in self._qvals.keys():
            for action in self._qvals[state].keys():
                yield state, action, self._qvals[state][action]

    def are_close(self, other: ActionValues) -> bool:
        if len(self) != len(other):
            return False
        for s, avals in self._qvals.items():
            for a, v1 in avals.items():
                v2 = other[s, a]
                if not math.isclose(v1, v2, rel_tol=0.001, abs_tol=0.001):
                    return False
        return True
