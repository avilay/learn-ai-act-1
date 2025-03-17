from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import math

from rltk.core import Action, Policy, State


class SimplePolicy(Policy):
    def __init__(self, rules: List[Tuple[State, Action, float]]) -> None:
        self._rules: Dict[State, Dict[Action, float]] = defaultdict(lambda: defaultdict(float))
        for s, a, p in rules:
            self._rules[s][a] = p
        for s in self._rules.keys():
            tot_prob = sum(self._rules[s].values())
            if not math.isclose(tot_prob, 1.0):
                raise ValueError(
                    f'Probabilities for states {s} do not sum to 1.0!')

    def prob(self, action: Action, given: State) -> float:
        if given not in self._rules:
            raise ValueError(f'No policy defined for state {given}!')
        return self._rules[given][action]

    def action(self, state: State) -> Action:
        if state not in self._rules:
            raise ValueError(f'No policy defined for state {state}!')
        actions_dist: Dict[Action, float] = self._rules[state]
        actions: List[Action] = []
        probs: List[float] = []
        for action, prob in actions_dist.items():
            actions.append(action)
            probs.append(prob)
        chosen_action: Action = np.random.choice(actions, p=probs)
        return chosen_action

    @classmethod
    def deterministic(cls, rules: List[Tuple[State, Action]]):
        rules2: List[Tuple[State, Action, float]] = []
        for s, a in rules:
            rules2.append((s, a, 1.))
        return cls(rules2)
