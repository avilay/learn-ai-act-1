import numpy as np
from .label_indexer import LabelIndexer

class QVals:
    def __init__(self, states, actions, random=True):
        self._states = LabelIndexer(states)
        self._actions = LabelIndexer(actions)
        shape = (len(self._states), len(self._actions))
        if random:
            self._qvals = np.random.random(shape)
        else:
            self._qvals = np.zeros(shape)

    def __getitem__(self, state_action):
        state, action = state_action
        return self._qvals[self._states[state], self._actions[action]]

    def __setitem__(self, state_action, val):
        state, action = state_action
        self._qvals[self._states[state], self._actions[action]] = val
