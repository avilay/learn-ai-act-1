from abc import ABC, abstractmethod
from typing import Tuple, Iterator

from .types import State


"""
Class to represent the state values of an environment following the Markov Decision Process.
"""


class StateValues(ABC):
    @abstractmethod
    def __getitem__(self, state: State) -> float:
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, state: State, value: float) -> None:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __add__(self, value: float) -> 'StateValues':
        """
        Add a constant value to these state values to create new StateValues.

        Args:
            value: The constant value to add to all the states.

        Returns:
            A new instance of StateValues with the constant added.
        """
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[State, float]]:
        raise NotImplemented()

    @abstractmethod
    def are_close(self, other: 'StateValues') -> bool:
        """
        Check if two instances of state values are close enough. Can be used to check for
        convergence between two instances of state values. It is expected that the concrete
        StateValues class will check the closeness for all possible states.

        Args:
            other: The other instance of StateValues.

        Returns:
            True if the all the states have values that are close enough, False otherwise.
        """
        raise NotImplementedError()
