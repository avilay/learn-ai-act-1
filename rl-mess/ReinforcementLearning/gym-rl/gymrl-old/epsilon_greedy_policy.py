import numpy as np
from typing import Sequence, Dict, Any
from .policy import Policy
from .q_table import QTable


class EpsilonGreedyPolicy(Policy):
    """
    Choose the best action with probability 1 - ε + ε/num_actions and every other action with
    probability ε/num_actions. For an unknown state, set any random action as the best action
    and then proceed to sample as before.
    """
    def __init__(self, qtab: QTable, ε: float) -> None:
        super().__init__(qtab)
        if ε < 0 or ε > 1:
            raise ValueError('ε must be between 0 and 1!')
        self._ε = ε
        self._cache: Dict[Any, np.ndarray] = {}

    def pmf(self, state: Any) -> Sequence[float]:
        if state not in self._cache:
            best_action = self._choose_best_action(state)
            probs = np.full(self._qtab.num_actions, self._ε/self._qtab.num_actions)
            probs[best_action] += 1 - self._ε
            self._cache[state] = probs
        return self._cache[state]
