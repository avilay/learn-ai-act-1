from collections import defaultdict
from copy import copy
from typing import Tuple, Any, Dict, Sequence, KeysView
import numpy as np


class QTable:
    """Table of action-values

    A table with states as rows and actions as columns. State can be any object. Actions are
    represented by integers. QTable does not really care about the semantic meaning of each int.

    Attributes:
        num_actions: The total number of possible actions in the environment. The actions will then
        be represented by ints 0 to num_actions-1.
    """
    def __init__(self, num_actions: int) -> None:
        self.num_actions = num_actions
        self._tab: Dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(num_actions))

    def __setitem__(self, state_action: Tuple[Any, int], value: float):
        """
        In the following example we are setting the action value of action 0 when in state 'some state'
        to the value 23.3.
        Example:
            qtab['some state', 0] = 23.3
        """
        state, action = state_action
        self._tab[state][action] = value

    def __getitem__(self, state_action: Tuple[Any, int]) -> float:
        """
        In the example below we are getting the action value of action 0 when in state 'somestate'.
        Example:
            value = qtab['somestate', 0]
        """
        state, action = state_action
        return self._tab[state][action]

    @property
    def states(self) -> KeysView[Any]:
        """
        Returns all the states that this QTable knows about.
        """
        return self._tab.keys()

    def all_values(self, state: Any) -> Sequence[float]:
        """
        Gets the action values of all actions when in the given state.
        Returns:
            A sequence of action values where the index represents the action and the value of
            the element represents the action value.
        """
        return copy(self._tab[state])

    def __str__(self):
        sb = ''
        for state, action_values in self._tab.items():
            sb += f'{state}: '
            for action_value in action_values:
                sb += f'{action_value:.3f} '
            sb += '\n'
        return sb

    def __repr__(self):
        return str(self._tab)
