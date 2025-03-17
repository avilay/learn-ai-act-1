from enum import Enum
from collections import namedtuple
import cv2

IMG_SIZE = 64


class Channel(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    ALL = 4

COLOR_SPACE = {
    'HSV': cv2.COLOR_RGB2HSV,
    'HLS': cv2.COLOR_RGB2HLS,
    'LUV': cv2.COLOR_RGB2LUV,
    'YUV': cv2.COLOR_RGB2YUV,
    'YCrCb': cv2.COLOR_RGB2YCrCb
}

HistogramSpec = namedtuple('HistogramSpec', ['channel', 'bins'])
HogSpec = namedtuple('HogSpec', ['channel', 'orientations', 'pixels_per_cell', 'cells_per_block'])
SpatialSpec = namedtuple('SpatialSpec', ['size'])


class FeatureSpec:
    def __init__(self, **kwargs):
        self.color_space = kwargs['color_space'] if 'color_space' in kwargs else 'RGB'
        self.spatial_spec = kwargs['spatial_spec'] if 'spatial_spec' in kwargs else self._default_spatial_spec()
        self.hist_spec = kwargs['hist_spec'] if 'hist_spec' in kwargs else self._default_hist_spec()
        self.hog_spec = kwargs['hog_spec'] if 'hog_spec' in kwargs else self._default_hog_spec()

    def _default_spatial_spec(self):
        return SpatialSpec(size=32)

    def _default_hist_spec(self):
        return HistogramSpec(channel=Channel.ALL, bins=32)

    def _default_hog_spec(self):
        return HogSpec(channel=Channel.ALL, orientations=9, pixels_per_cell=8, cells_per_block=2)

    def __repr__(self):
        ret = f'color_space={self.color_space}\n'
        if self.spatial_spec:
            ret += f'spatial_spec=(size={self.spatial_spec.size})\n'
        else:
            ret += 'spatial_spec=False\n'

        if self.hist_spec:
            ret += f'histogram_spec=(channel={self.hist_spec.channel}, bin={self.hist_spec.bins})\n'
        else:
            ret += 'histogram_spec=False\n'

        if self.hog_spec:
            ret += f'hog_spec=(channel={self.hog_spec.channel}, orientations={self.hog_spec.orientations}, pixels_per_cell={self.hog_spec.pixels_per_cell}, cells_per_block={self.hog_spec.cells_per_block})'
        else:
            ret += 'hog_spec=False\n'

        return ret
