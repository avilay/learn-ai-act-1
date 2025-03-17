from rltk.core import Action


class GridAction(Action):
    _up = None
    _down = None
    _left = None
    _right = None

    def __init__(self, a: int) -> None:
        self._a = a

    def __repr__(self):
        if self._a == 1:
            return 'UP'
        elif self._a == 2:
            return 'DOWN'
        elif self._a == 3:
            return 'LEFT'
        elif self._a == 4:
            return 'RIGHT'
        else:
            raise RuntimeError('KABOOM!')

    def __hash__(self):
        return hash(self._a)

    def __eq__(self, other):
        return self._a == other._a  # noqa

    @classmethod
    def up(cls):
        if cls._up is None:
            cls._up = GridAction(1)
        return cls._up

    @classmethod
    def down(cls):
        if cls._down is None:
            cls._down = GridAction(2)
        return cls._down

    @classmethod
    def left(cls):
        if cls._left is None:
            cls._left = GridAction(3)
        return cls._left

    @classmethod
    def right(cls):
        if cls._right is None:
            cls._right = GridAction(4)
        return cls._right
