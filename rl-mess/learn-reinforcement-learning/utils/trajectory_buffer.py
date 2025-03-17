from .buffer import Buffer
from .step import Step
from .trajectory import Trajectory


class TrajectoryBuffer(Buffer):
    def __init__(self, env, capacity=1000, avg_over=20, discount_factor=0.99):
        super().__init__(env, capacity, avg_over)
        self._discount_factor = discount_factor

    def replinish(self, policy, fraction):
        if fraction <= 0 or fraction > 1:
            raise ValueError("Fraction must be between (0, 1]!")
        num_trajectories = int(self._capacity * fraction)
        for _ in range(num_trajectories):
            t = Trajectory(discount_factor=self._discount_factor)
            done = False
            state = self._env.reset()
            while not done:
                action = policy(state)
                next_state, reward, done, _ = self._env.step(action)
                s = Step(
                    state=state, action=action, reward=reward, next_state=next_state, done=done
                )
                t.add_step(s)
                state = next_state
            t.calculate_returns()
            self._scores.append(t.total_score)
            self._buffer.append(t)
