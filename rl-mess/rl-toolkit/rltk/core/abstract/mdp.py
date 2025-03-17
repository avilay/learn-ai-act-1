from abc import ABC, abstractmethod

from .types import State, Action

"""
Class representing a Markov Decision Process.
"""


class MDP(ABC):
    @abstractmethod
    def reward(self, state: State, action: Action, next_state: State) -> float:
        """
        Gets the reward for taking action in the current state and landing in the next_state. It is
        possible that the MDP does not give out reward based on the next_state, but only on the
        action taken in the current state, in which case the concrete implementation of this class
        can simply ignore the next_state.

        Args:
            state: The current state.
            action: The action taken in the current state.
            next_state: The state that the agent will be in after it has taken the action.

        Returns:
            Any real number.
        """
        raise NotImplementedError()

    @abstractmethod
    def trans_prob(self, state: State, action: Action, next_state: State) -> float:
        """
        Gets the transition probability of transtioning from current state to the next state
        if the specified action is taken -
        P(s_|s, a)
        For stochastic MDPs this will be distribution of next states given the current state and
        action. For deterministic MDPs, this will be 1. for the next state and 0. for all other states.

        Args:
            state: The current state.
            action: The action being taken in the current state.
            next_state: The state whose transition probability is being queried.

        Returns:
            A real number between [0, 1].
        """
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """
        Checks whether a state is terminal or not.
        """
        raise NotImplementedError()
