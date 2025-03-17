from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Any, Sequence
from .q_table import QTable
import numpy as np

class Policy(ABC):
    """
    Abstract class representing the policy followed by an agent. Subclasses must implement
    the pmf method.
    """
    def __init__(self, qtab: QTable) -> None:
        self._qtab = deepcopy(qtab)
        self._cache: Dict[Any, np.ndarray] = {}

    def action(self, state: Any) -> int:
        """
        Samples the action from the probability distribution of possible actions in the given
        state.
        """
        return np.random.choice(np.arange(self._qtab.num_actions), p=self.pmf(state))

    def prob(self, state: Any, action: int) -> float:
        """
        Gets the probability of taking the specified action when in the given state.
        P(A=action | S=state)
        """
        return self.pmf(state)[action]

    def _choose_best_action(self, state: Any) -> int:
        """
        The best action is the one with the maximum value. It is possible that there are multiple
        actions all with the same max value. In that case, will randomly choose an action from
        amongst these.
        """
        # If there are multiple best actions, choose one at random
        vals = self._qtab.all_values(state)
        best_actions = np.argwhere(vals == np.max(vals)).flatten()
        best_action = np.random.choice(best_actions)
        return best_action

    def __repr__(self):
        sb = ''
        for state in self._qtab.states:
            action = self.action(state)
            sb += f'{state}={action}\n'
        return sb

    @abstractmethod
    def pmf(self, state: Any) -> Sequence[float]:
        """
        Get the probability mass function of all possible actions when in the given state.
        Return:
            A sequence of probabilities (between 0 and 1) where the index represents the probability
            of that action.
        """
        pass
