import numpy as np


class Step:
    def __init__(self, **kwargs):
        self.state = kwargs.get("state")
        self.action = kwargs.get("action")
        self.reward = kwargs.get("reward")
        self.next_state = kwargs.get("next_state")
        self.done = kwargs.get("done", False)

    def __eq__(self, other):
        return (
            isinstance(other, Step)
            and other.state == self.state
            and other.action == self.action
            and other.next_state == self.next_state
            and other.done == self.done
        )

    def __hash__(self):
        return hash(self.state, self.action, self.reward, self.next_state, self.done)

    @staticmethod
    def stripe(steps):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for step in steps:
            states.append(step.state)
            actions.append(step.action)
            rewards.append(step.reward)
            next_states.append(step.next_state)
            dones.append(step.done)
        return dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states),
            dones=np.array(dones, np.int),
        )
