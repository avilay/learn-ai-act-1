from abc import ABC, abstractmethod
from .types import State


class StateValues(ABC):
    @abstractmethod
    def __getitem__(self, state: State) -> float:
        pass

    @abstractmethod
    def __setitem__(self, state: State, value: float) -> None:
        pass

    @abstractmethod
    def __add__(self, value: float) -> StateValues:  # noqa
        pass

    @abstractmethod
    def are_close(self, svals: StateValues) -> bool:  # noqa
        pass

    @abstractmethod
    @classmethod
    def zeros(cls) -> StateValues:  # noqa
        pass

    @abstractmethod
    @classmethod
    def random(cls) -> StateValues:  # noqa
        pass
