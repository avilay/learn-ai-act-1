from .abstract.types import State, Action


class Step:
    def __init__(self, **kwargs) -> None:
        self.state: State = kwargs['state']
        self.action: Action = kwargs['action']
        self.reward: float = kwargs['reward']
        self.next_state: State = kwargs['next_state']

    def __str__(self) -> str:
        return f'<s={self.state}, a={self.action}, r={self.reward}, s_={self.next_state}>'

    def __eq__(self, other) -> bool:
        return self.state == other.state and \
            self.action == other.action and \
            self.reward == other.reward and \
            self.next_state == other.next_state
