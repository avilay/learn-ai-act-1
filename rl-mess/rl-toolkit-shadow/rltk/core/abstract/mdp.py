from abc import ABC, abstractmethod
from .types import State, Action

class MDP(ABC):
    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> float:
        pass

    @abstractmethod
    def trans_prob(self, state: State, action: Action, next_state: State) -> float:
        pass
