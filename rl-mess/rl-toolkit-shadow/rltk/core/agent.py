from .policy import Policy
from .env import Env
from .episode import Episode
from .step import Step
from .abstract.state import State
from .abstract.action import Action
from uuid import UUID, uuid4
from typing import Iterable


class Agent:
    def __init__(self, env: Env):
        self._env = env
        self._id: UUID = uuid4()
        self._t = 0

    def move_to_end(self, pi: Policy) -> Episode:
        episode = Episode()
        for step in self.move(pi):
            episode.append(step)
        return episode

    def move(self, pi: Policy) -> Iterable[Step]:
        s: State = self._env.curr_state(self._id)
        if self._env.is_terminal(s):
            raise StopIteration()

        self._t += 1
        a: Action = pi.action(s)
        r, s_ = self._env.move(self._id, a)
        yield Step(t=self._t, s=s, a=a, r=r, s_=s_)
