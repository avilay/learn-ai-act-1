from abc import ABC, abstractmethod
from typing import Set, Tuple, Sequence

from .types import State, Action, Reward


class DiscreteMDP(ABC):
    @property
    @abstractmethod
    def states(self) -> Set[State]:
        pass

    @property
    @abstractmethod
    def actions(self) -> Set[State]:
        pass

    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> Reward:
        pass

    @abstractmethod
    def dist(self, state: State, action: Action) -> Sequence[Tuple[State, float]]:
        pass

    @abstractmethod
    def is_terminal(self, state: State):
        pass
