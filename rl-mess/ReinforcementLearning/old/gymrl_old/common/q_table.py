import numpy as np
from typing import Tuple, Any, Dict
from collections import defaultdict
from .policy import Policy


class QTable:
    def __init__(self, num_actions: int) -> None:
        self._tab: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(num_actions))
        self.num_actions = num_actions

    def __getitem__(self, state_action: Tuple[Any, int]) -> float:
        state, action = state_action
        return self._tab[state][action]

    def __setitem__(self, state_action: Tuple[Any, int], value: float):
        state, action = state_action
        self._tab[state][action] = value

    def greedy_policy(self) -> Policy:
        pass

    def epsilon_greedy_policy(self, ε: float) -> Policy:
        pass
