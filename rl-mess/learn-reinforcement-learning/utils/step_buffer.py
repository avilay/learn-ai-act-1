import numpy as np

from .buffer import Buffer
from .step import Step


class StepBuffer(Buffer):
    def __init__(self, env, capacity=int(1e5), avg_over=20):
        super().__init__(env, capacity, avg_over)

    def replinish(self, policy, fraction):
        if fraction <= 0 or fraction > 1:
            raise ValueError("Fraction must be between (0, 1]!")
        num_steps = int(self._capacity * fraction)
        state = self._env.reset()
        score = 0
        for _ in range(num_steps):
            action = policy(state)
            next_state, reward, done, _ = self._env.step(action)
            self._buffer.append(
                Step(state=state, action=action, reward=reward, next_state=next_state, done=done)
            )
            score += reward
            if done:
                state = self._env.reset()
                self._scores.append(score)
                score = 0
            else:
                state = next_state
