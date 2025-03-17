from typing import Tuple
from uuid import UUID
from .state import State
from .action import Action
from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def curr_state(self, agent_id: UUID) -> State:
        raise NotImplementedError()

    @abstractmethod
    def move(self, agent_id: UUID, action: Action) -> Tuple[float, State]:
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        raise NotImplementedError()
