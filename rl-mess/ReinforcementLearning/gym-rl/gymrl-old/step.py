from typing import Any

class Step:
    """Represents a single time step in an episode

    Consider an episode like -
    S0, A0, R1, S1, A1, R2, S2, A2, ..., Rt, St, At, R{t+1}, ...
    Where an agent starts out in state S0, takes an action A0, gets a reward R1 and lands in state
    S1 and so on.
    Then the step object represents -
    [S0, A0], [R1, S1, A1], [R2, S2, A2], ..., [Rt, St, At], ...

    Attributes:
        reward: The reward for the action taken in the previous time step.
        state: The state in which the agent finds itself in the current time step.
        action: The action that the agent takes in the current time step.
    """
    def __init__(self, **kwargs):
        self.reward: float = kwargs.get('reward')
        self.state: Any = kwargs.get('state')
        self.action: int = kwargs.get('action')

    def __repr__(self):
        return f'<Step(reward={self.reward} state={self.state} action={self.action})>'
