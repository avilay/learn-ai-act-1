from typing import List, Tuple
from abc import ABC, abstractmethod
from .types import State, Action
from .action_values import ActionValues


class Policy(ABC):
    @abstractmethod
    def prob(self, action: Action, given: State) -> float:
        pass

    @abstractmethod
    def action(self, state: State) -> Action:
        pass

    @classmethod
    @abstractmethod
    def greedy(cls, qvals: ActionValues) -> Policy:  # noqa
        pass

    @classmethod
    @abstractmethod
    def epsilon_greedy(cls, qvals: ActionValues, epsilon = 0.01) -> Policy:  # noqa
        pass

    @classmethod
    @abstractmethod
    def deterministic(cls, rules: List[Tuple[State, Action]]) -> Policy:  # noqa
        pass
