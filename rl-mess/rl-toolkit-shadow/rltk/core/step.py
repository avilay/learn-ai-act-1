from .abstract.state import State
from .abstract.action import Action


class Step:
    def __init__(self, **kwargs):
        self.t: int = kwargs.get('t')
        self.s: State = kwargs.get('s')
        self.a: Action = kwargs.get('a')
        self.r: float = kwargs.get('r')
        self.s_: State = kwargs.get('s_')
step