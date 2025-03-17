from .step import Step
from typing import List, Iterator


class Episode:
    def __init__(self) -> None:
        self._steps: List[Step] = []
        self._cum_rewards: List[float] = []
        self._num_steps = 0

    def append(self, step: Step) -> None:
        if step is None:
            raise ValueError('Cannot add a None step!')
        self._steps.append(step)

    def calc_cum_rewards(self, gamma=1.):
        if self._num_steps == len(self._steps):
            return
        self._cum_rewards: List[float] = [0] * len(self._steps)
        self._cum_rewards[-1] = self._steps[-1].reward
        for i in range(len(self._steps)-2, -1, -1):
            self._cum_rewards[i] = self._steps[i].reward + gamma * self._cum_rewards[i+1]
        self._num_steps = len(self._steps)

    def __iter__(self) -> Iterator[Step]:
        for step in self._steps:
            yield step

    def __len__(self) -> int:
        return len(self._steps)

    def cum_reward(self, t: int) -> float:
        if self._num_steps != len(self._steps):
            raise RuntimeError('cumulative rewards have not been calculated!')
        return self._cum_rewards[t]
