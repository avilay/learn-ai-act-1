from copy import copy
from typing import Set, Dict, Tuple, Mapping, List, Sequence, Callable

from .types import State, Action, Reward


class MDP:
    def __init__(self,
        states: Set[State],
        actions: Set[State],
        rewards: Callable[[State, Action, State], float],
        probs: Dict[Tuple[State, Action], List[Tuple[State, float]]],
        terminal_states: Set[State]) -> None:
        self._states = states
        self._actions = actions
        self._rewards = rewards
        self._probs = probs
        self._terminal_states = terminal_states

    @property
    def states(self) -> Set[State]:
        return copy(self._states)

    @property
    def actions(self) -> Set[State]:
        return copy(self._actions)

    def reward(self, state: State, action: Action, next_state: State) -> Reward:
        return self._rewards(state, action, next_state)

    def dist(self, state: State, action: Action) -> Sequence[Tuple[State, float]]:
        return self._probs[(state, action)]

    def is_terminal(self, state: State):
        return state in self._terminal_states
