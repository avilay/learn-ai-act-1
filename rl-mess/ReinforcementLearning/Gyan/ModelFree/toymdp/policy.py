import numpy as np
from typing import Mapping, Dict, List, Tuple
from collections import defaultdict
from .types import State, Action, Reward


class Policy:
    def __init__(self, rules: Dict[State, List[Tuple[Action, float]]]) -> None:
        self._rules = rules

    def __call__(self, state: State) -> float:
        possible_actions = self._rules[state]
        chosen_action = np.random.choice(possible_actions, p=[x[1] for x in possible_actions])
        return chosen_action[0]

    def dist(self, state: State) -> Mapping[Action, float]:
        possible_actions = self._rules[state]
        pd: Dict[Action, float] = defaultdict(float)
        for action, prob in possible_actions:
            pd[action] = prob
        return pd
