import numpy as np
from typing import List, Sequence, Tuple, Set
from .types import State, Action, Reward
from .mdp import MDP
from .policy import Policy


class Simulator:
    def __init__(self, mdp: MDP, policy: Policy, terminal_states: Set[State]) -> None:
        self._mdp = mdp
        self._policy = policy
        self._terminal_states = terminal_states

    @property
    def state(self):
        return self._curr_state

    @state.setter
    def state(self, s :State):
        self._curr_state = s

    def step(self):
        if self._curr_state in self._terminal_states:
            raise StopIteration()
        action = self._policy(self._curr_state)
        next_states: Sequence[Tuple[State, float]] = self._mdp.dist(self._curr_state, action)
        chosen_next_state = np.random.choice(next_states, p=[x[1] for x in next_states])
        reward = self._mdp.reward(self._curr_state, action, chosen_next_state[0])
        self._curr_state = chosen_next_state[0]
        yield reward