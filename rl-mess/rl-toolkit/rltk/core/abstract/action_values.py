from abc import ABC, abstractmethod
from typing import Tuple, Iterator

from .types import State, Action

"""
Class to represent the action values or Q values of an environment following the Markov Decision Process.
"""


class ActionValues(ABC):
    @abstractmethod
    def __getitem__(self, key: Tuple[State, Action]) -> float:
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, key: Tuple[State, Action], value: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[State, Action, float]]:
        raise NotImplemented()

    @abstractmethod
    def __add__(self, value: float) -> 'ActionValues':
        """
        Add a constant value to these action values to create new ActionValues.

        Args:
            value: The constant value to add to all the state-action pairs.

        Returns:
            A new instance of ActionValues with the constant added.
        """
        raise NotImplementedError()

    @abstractmethod
    def are_close(self, other: 'ActionValues') -> bool:
        """
        Check if two instances of action values are close enough. Can be used to check for
        convergence between two instances of action values. It is expected that the concrete
        ActionValues class will check the closeness for all possible state-action combinations.

        Args:
            other: The other instance of ActionValues.

        Returns:
            True if all the state-actions have values that are close enough, False otherwise.
        """
        raise NotImplementedError()
