from typing import Dict, List, Tuple
from uuid import UUID

import numpy as np

from rltk.core import Action, Env, State, DiscreteMDP
from rltk.core.errors import AgentTerminatedError, DuplicateAgentError, UnknownAgentError


class SimpleEnv(Env):
    def __init__(self, mdp: DiscreteMDP, default_start_state: State) -> None:
        self._default_start_state = default_start_state
        self._mdp = mdp
        self._agents: Dict[UUID, State] = {}

    def enter(self, agent_id: UUID) -> State:
        if agent_id in self._agents:
            raise DuplicateAgentError(f'{agent_id} is already in the env!')
        self._agents[agent_id] = self._default_start_state
        return self._default_start_state

    def curr_state(self, agent_id: UUID) -> State:
        if agent_id not in self._agents:
            raise UnknownAgentError(f'{agent_id} has not entered the env!')
        return self._agents[agent_id]

    def is_terminal(self, state: State) -> bool:
        return self._mdp.is_terminal(state)

    def move(self, agent_id: UUID, action: Action) -> Tuple[float, State]:
        if agent_id not in self._agents:
            raise UnknownAgentError(f'{agent_id} has not entered the env!')
        next_states: List[State] = []
        trans_probs: List[float] = []
        s = self._agents[agent_id]
        if self._mdp.is_terminal(s):
            raise AgentTerminatedError(
                f'{agent_id} has already been terminated!')
        a = action
        for s_ in self._mdp.states:
            p = self._mdp.trans_prob(s, a, s_)
            next_states.append(s_)
            trans_probs.append(p)
        idx = np.random.choice(list(range(len(next_states))), p=trans_probs)
        s_ = next_states[idx]
        self._agents[agent_id] = s_

        r = self._mdp.reward(s, a, s_)

        return r, s_
