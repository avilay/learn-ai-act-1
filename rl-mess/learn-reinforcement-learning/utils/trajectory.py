import numpy as np


class Trajectory:
    def __init__(self, discount_factor=0.99):
        self._steps = []
        self._returns = []
        self._discount_factor = discount_factor
        self._total_score = 0

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False

        if len(self._steps) != len(other._steps):
            return False

        for self_step, other_step in zip(self._steps, other._steps):
            if self_step != other_step:
                return False

        return True

    def __hash__(self):
        return hash(tuple([hash(s) for s in self._steps]))

    @property
    def total_score(self):
        return self._total_score

    def add_step(self, step):
        self._steps.append(step)
        self._total_score += step.reward

    def calculate_returns(self):
        self._returns = [None] * len(self._steps)
        self._returns[-1] = self._steps[-1].reward
        for i in range(len(self._steps) - 2, -1, -1):
            self._returns[i] = self._steps[i].reward + self._discount_factor * self._returns[i + 1]
        return self._returns

    def __len__(self):
        return len(self._steps)

    def __getitem__(self, idx):
        step = self._steps[idx]
        return_ = self._returns[idx] if self._returns else None
        return (step, return_)

    @staticmethod
    def stripe(trajectories):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        returns = []
        for trajectory in trajectories:
            for step, g in zip(trajectory._steps, trajectory._returns):
                states.append(step.state)
                actions.append(step.action)
                rewards.append(step.reward)
                next_states.append(step.next_state)
                dones.append(step.done)
                returns.append(g)
        return dict(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states),
            dones=np.array(dones, np.int),
            returns=np.array(returns),
        )
