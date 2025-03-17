from collections import namedtuple
import numpy as np

Point = namedtuple('Point', ['x', 'y'])


class Box:
    def __init__(self, top_left=None, bottom_right=None):
        self.top_left = top_left
        self.bottom_right = bottom_right

    @staticmethod
    def create(center, width, height):
        top_left = Point(
            x=center.x - width // 2,
            y=center.y - height // 2
        )
        bottom_right = Point(
            x=center.x + width // 2,
            y=center.y + height // 2
        )
        return Box(top_left, bottom_right)

    @property
    def center(self):
        x = self.top_left.x + (self.bottom_right.x - self.top_left.x) // 2
        y = self.top_left.y + (self.bottom_right.y - self.top_left.y) // 2
        return Point(x=x, y=y)

    @property
    def width(self):
        return self.bottom_right.x - self.top_left.x

    @property
    def height(self):
        return self.bottom_right.y - self.top_left.y

    def distance(self, box):
        ypart = (box.center.y - self.center.y) ** 2
        xpart = (box.center.x - self.center.x) ** 2
        return np.sqrt(xpart + ypart)

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return f'Top Left: x={self.top_left.x} y={self.top_left.y}  Bottom Right: x={self.bottom_right.x} y={self.bottom_right.y}'
