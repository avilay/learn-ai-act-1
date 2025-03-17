from typing import Any, Sequence
import numpy as np
from .policy import Policy

class GreedyPolicy(Policy):
    """
    Always choose the best action, i.e., the one with the maximum value.
    When there are multiple best actions choose one randomly. For an unknown state,
    i.e., a state that is not in the QTable, choose any action randomly.
    """
    def pmf(self, state: Any) -> Sequence[float]:
        if state not in self._cache:
            best_action = self._choose_best_action(state)
            probs = np.zeros(self._qtab.num_actions)
            probs[best_action] = 1.
            self._cache[state] = probs
        return self._cache[state]

