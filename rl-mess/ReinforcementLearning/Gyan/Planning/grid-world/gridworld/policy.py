from typing import Mapping, Callable
import numpy as np
from .world import Action, State


class Policy:
    def __init__(self, pmf: Callable[[State], Mapping[Action, float]]) -> None:
        self.pmf = pmf

    def __call__(self, s: State) -> Action:
        action_dist = self.pmf(s)
        actions = []
        probs = []
        for a, p in action_dist.items():
            actions.append(a)
            probs.append(p)
        return np.random.choice(actions, p=probs)
