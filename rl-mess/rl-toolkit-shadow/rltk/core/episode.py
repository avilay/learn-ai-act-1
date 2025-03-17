from typing import List, Iterator
from .step import Step


class Episode:
    def __init__(self) -> None:
        self._steps: List[Step] = []
        self._cum_rewards: List[float] = []

    def append(self, step: Step) -> None:
        self._steps.append(step)

    def calc_cum_rewards(self):
        self._cum_rewards: List[float] = [0] * len(self._steps)
        self._cum_rewards[-1] = self._steps[-1].r
        for i in range(len(self._steps)-2, -1, -1):
            self._cum_rewards[i] = self._steps[i].r + self._cum_rewards[i+1]

    def __iter__(self) -> Iterator[Step]:
        for step in self._steps:
            yield step

    def cum_reward(self, t: int) -> float:
        return self._cum_rewards[t]
