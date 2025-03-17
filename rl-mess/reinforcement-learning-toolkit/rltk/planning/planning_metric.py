from typing import Dict, Mapping, Set

from rltk.core.types import Action, State


class PlanningMetric:
    def __init__(self, context_id: str, iter_num: int, **kwargs):
        self.context_id = context_id
        self.iter_num = iter_num
        self.svals: Dict[State, float] = kwargs.get('svals', {})
        self.qvals: Dict[State, Mapping[Action, float]
                         ] = kwargs.get('qvals', {})
        self.pidist: Dict[State, Mapping[Action, float]] = kwargs.get('pidist', {})

    def __str__(self) -> str:
        states: Set[State] = set(self.svals.keys())
        states = states.union(set(self.qvals.keys()))
        states = states.union(set(self.pidist.keys()))

        actions: Set[Action] = set()
        for state in states:
            actions = actions.union(set(self.qvals.get(state, {}).keys()))
            actions = actions.union(set(self.pidist.get(state, {}).keys()))

        ret = f'{self.context_id} - iter: {self.iter_num}\n'
        for state in states:
            sval = self.svals.get(state, float('nan'))
            ret += f'{state} = {sval:.5f}\n'
            for action in actions:
                qval = self.qvals.get(state, {}).get(action, float('nan'))
                prob = self.pidist.get(state, {}).get(action, float('nan'))
                ret += f'\t({state} {action})={qval:.5f}  P({action}|{state})={prob:.5f}\n'
        return ret
