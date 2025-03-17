from abc import ABC, abstractmethod
from typing import List, Tuple

from .types import State, Action

"""
Class to represent a stochastic policy for operating in an environment following a Markov Decision Process.
"""


class Policy(ABC):
    @abstractmethod
    def prob(self, action: Action, given: State) -> float:
        """
        The probability of taking the specified action in the given state.

        Args:
            action: The action whose probability is being queried.
            given: The state in which this action will be taken.

        Returns:
            Probabilty of taking the action in the given state. This must be between [0, 1].
        """
        raise NotImplementedError()

    @abstractmethod
    def action(self, state: State) -> Action:
        """
        Samples the action to be taken in the specified state based on the probability distribution
        of actions in the given state.

        Args:
            state: The current state of the agent.

        Returns:
            The action that is sampled from the probability distribution of actions.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def deterministic(cls, rules: List[Tuple[State, Action]]) -> 'Policy':
        """
        Generate a simple deterministic policy based on the specified states and the corresponding
        actions to take in that state.

        Args:
            rules: A list of state, action pairs where π(a|s) = 1. if (s, a) show up in the list.

        Returns:
            A new Policy which follows the specified rules.
        """
        raise NotImplementedError()
