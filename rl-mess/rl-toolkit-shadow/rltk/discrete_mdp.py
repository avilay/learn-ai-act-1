from typing import Iterator
from abc import abstractmethod
from rltk.core import MDP, State, Action


class DiscreteMDP(MDP):
    @property
    @abstractmethod
    def states(self) -> Iterator[State]:
        pass

    @property
    @abstractmethod
    def actions(self) -> Iterator[Action]:
        pass
