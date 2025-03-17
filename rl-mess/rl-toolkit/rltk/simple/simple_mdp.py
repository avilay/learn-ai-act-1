from copy import deepcopy
from typing import Iterator, Sequence, Tuple, Dict, Callable

from rltk.core import Action, State, DiscreteMDP, MdpFunctionError

RewardFunc = Callable[[State, Action, State], float]


class SimpleMDP(DiscreteMDP):
    def __init__(
        self,
        states: Sequence[State],
        actions: Sequence[Action],
        terminal_states: Sequence[State],
        probs: Dict[Tuple[State, Action], Dict[State, float]],
        reward: RewardFunc,
    ) -> None:
        """
        Defines a simple MDP with discrete and countable states and actions.

        Args:
            states: A set of State objects that is the full and exhaustive list of possible states
            in this MDP.
            actions: A set of Action objects that is the full and exhaustive list of possible
            actions in this MDP.
            terminal_states: A set of State objects that are terminal. This must be a subset of states.

        """
        self._states = set(states)
        self._actions = set(actions)
        self._terminal_states = set(terminal_states)
        if not self._terminal_states.issubset(self._states):
            raise ValueError('Terminal states must be a subset of states!')

        if not reward or not probs:
            raise ValueError('Must specify both rewards and probs!')
        for s, a in probs.keys():
            next_state_dist = probs[(s,a)]
            if sum(next_state_dist.values()) != 1.:
                raise ValueError(f'Transitional probabilities for {s}{a} is not summing upto 1!')
            for p in next_state_dist.values():
                if p < 0. or p > 1.:
                    raise ValueError('One of the transitional probability for {s}{a} is not valid!')
        self._probs = deepcopy(probs)
        self._reward = reward

    def reward(self, state: State, action: Action, next_state: State) -> float:
        if self.is_terminal(state):
            raise ValueError(f'Cannot transition from terminal state {state}!')
        try:
            return self._reward(state, action, next_state)
        except Exception as e:
            raise MdpFunctionError(f'Unable to call reward({state},{action},{next_state})!') from e

    def trans_prob(self, state: State, action: Action, next_state: State) -> float:
        if self.is_terminal(state):
                raise ValueError(f'Cannot transition from terminal state {state}!')
        if (state, action) not in self._probs:
            return 0.
        if next_state not in self._probs[(state, action)]:
            return 0.
        return self._probs[(state, action)][next_state]

    @property
    def states(self) -> Iterator[State]:
        return iter(self._states)

    @property
    def actions(self) -> Iterator[Action]:
        return iter(self._actions)

    def is_terminal(self, state: State) -> bool:
        return state in self._terminal_states
