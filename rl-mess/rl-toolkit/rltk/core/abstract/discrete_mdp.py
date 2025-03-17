from rltk.core import MDP, State, Action
from abc import abstractmethod
from typing import Iterator

"""
Class representing an MDP with discrete and countable states and actions.
"""


class DiscreteMDP(MDP):
    @property
    @abstractmethod
    def states(self) -> Iterator[State]:
        """
        Iterate through all the states of the MDP. This must be a finite iterator.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def actions(self) -> Iterator[Action]:
        """
        Iterate through all the actions of the MDP. This must by a finite iterator.
        """
        pass
