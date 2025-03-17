from typing import Iterator, List, Optional, Tuple

from .step import Step


class Episode:
    def __init__(self) -> None:
        self._steps: List[Step] = []
        self._returns: List[float] = []
        self._num_steps = 0

    def append(self, step: Step) -> None:
        if step is None:
            raise ValueError("Cannot add a None step!")
        self._steps.append(step)

    def calc_returns(self, gamma=1.0):
        if self._num_steps == len(self._steps):
            return
        self._returns: List[float] = [0.0] * len(self._steps)
        self._returns[-1] = self._steps[-1].reward
        for i in range(len(self._steps) - 2, -1, -1):
            self._returns[i] = self._steps[i].reward + gamma * self._returns[i + 1]
        self._num_steps = len(self._steps)

    def __iter__(self) -> Iterator[Tuple[Step, Optional[float]]]:
        if self._num_steps != len(self._steps):
            returns = [None] * len(self._steps)
        else:
            returns = self._returns

        for step, return_ in zip(self._steps, returns):
            yield step, return_

    def __getitem__(self, idx: int) -> Tuple[Step, Optional[float]]:
        if self._num_steps != len(self._steps):
            g = None
        else:
            g = self._returns[idx]
        return self._steps[idx], g

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self):
        if self._num_steps != len(self._steps):
            self.calc_returns()
        ret = "Episode -\n"
        for step, return_ in zip(self._steps, self._returns):
            ret += f"\t{step} {return_}\n"
        return ret
