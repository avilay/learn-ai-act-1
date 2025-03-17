from .episode import Episode
from .abstract.env import Env
from uuid import UUID, uuid4
from .abstract.policy import Policy
from .abstract.types import State, Action
from .step import Step


class Agent:
    def __init__(self, env: Env) -> None:
        self._env = env
        self._id: UUID = uuid4()
        self._t = 0
        self._env.enter(self._id)

    def move_to_end(self, pi: Policy) -> Episode:
        episode = Episode()
        s_: State = self._env.curr_state(self._id)
        while not self._env.is_terminal(s_):
            step = self.move(pi)
            episode.append(step)
            s_ = step.next_state
        return episode

    def move(self, pi: Policy) -> Step:
        s: State = self._env.curr_state(self._id)
        if self._env.is_terminal(s):
            raise ValueError()
        self._t += 1
        a: Action = pi.action(s)
        r, s_ = self._env.move(self._id, a)
        return Step(
            state=s,
            action=a,
            reward=r,
            next_state=s_
        )
