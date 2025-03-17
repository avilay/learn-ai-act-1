import numpy as np
from typing import List, Tuple, Set, Union, cast
from .types import State, Action, QVals
from rltk.label_indexer import LabelIndexer


# class Policy:
#     def __init__(self, rules: Dict[State, List[Tuple[Action, float]]]) -> None:
#         self._rules = rules

#     def __call__(self, state: State) -> Action:
#         possible_actions = self._rules[state]
#         chosen_action = np.random.choice(
#             possible_actions, p=[x[1] for x in possible_actions])
#         return chosen_action[0]

#     def dist(self, state: State) -> Mapping[Action, float]:
#         possible_actions = self._rules[state]
#         pd: Dict[Action, float] = defaultdict(float)
#         for action, prob in possible_actions:
#             pd[action] = prob
#         return pd

class Policy:
    def __init__(self, rules: List[Tuple[State, Action, float]]) -> None:
        uniq_states: Set[State] = set()
        uniq_actions: Set[Action] = set()
        for state, action, _ in rules:
            uniq_states.add(state)
            uniq_actions.add(action)
        self._states = LabelIndexer(uniq_states)
        self._actions = LabelIndexer(uniq_actions)

        self._rules = np.zeros((len(self._states), len(self._actions)))

    def __call__(self, state: State, action: Action = None) -> Union[float, Action]:
        if action is None:
            return self.sample(state)

        if state in self._states and action in self._actions:
            s = self._states[state]
            a = self._actions[action]
            return self._rules[s, a]
        else:
            return 0.

    def sample(self, state: State) -> Action:
        if state in self._states:
            s = self._states[state]
            actions = self._rules[s]
            idxs = list(range(len(actions)))
            chosen_action = np.random.choice(idxs, p=actions)
            return cast(str, self._actions[chosen_action])
        else:
            raise ValueError()


def create_greedy(qvals: QVals) -> Policy:
    rules = []
    for state in qvals.states():
        actions, vals = qvals.qvals(state)
        max_val_idx = np.argmax(vals)
        best_action = actions[max_val_idx]
        rules.append((state, best_action, 1.))
    return Policy(rules)


def create_epsilon_greedy(cls, qvals: QVals, epsilon=0.01) -> Policy:
    raise NotImplementedError()


def create_deterministic(cls, policy: List[Tuple[State, Action]]) -> Policy:
    rules = []
    for state, action in policy:
        rules.append((state, action, 1.))
    return Policy(rules)
