from typing import Tuple
from abc import ABC, abstractmethod
from .types import State, Action


class ActionValues(ABC):
    @abstractmethod
    def __getitem__(self, key: Tuple[State, Action]) -> float:
        pass

    @abstractmethod
    def __setitem__(Self, key: Tuple[State, Action], value: float) -> None:
        pass

    @abstractmethod
    def __add__(self, value: float) -> None:
        pass

    @abstractmethod
    def are_close(self, qvals: ActionValues) -> bool:  # noqa
        pass

    @abstractmethod
    @classmethod
    def zeros(cls) -> ActionValues:  # noqa
        pass

    @abstractmethod
    @classmethod
    def random(cls) -> ActionValues: # noqa
        pass
