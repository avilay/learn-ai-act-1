import numpy as np
from scipy.ndimage.measurements import label

from ..features.box import Box, Point


class FrameCache:
    def __init__(self, debug=False):
        self._heat_frames = []
        self._wtd_frames = []

    def heat_maps(self):
        if self._wtd_frames:
            candidates = self._heat_frames[-11:]
            wtd_frame = self._wtd_frames[-1]
            return candidates, wtd_frame
        else:
            return [], None

    def add(self, frame, hot_windows):
        heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
        for hot_window in hot_windows:
            row_start = hot_window.top_left.y
            row_end = hot_window.bottom_right.y + 1

            col_start = hot_window.top_left.x
            col_end = hot_window.bottom_right.x + 1

            heat[row_start:row_end, col_start:col_end] += 1
        self._heat_frames.append(heat)

    def car_boxes(self):
        if len(self._heat_frames) < 12:
            return []

        candidates = self._heat_frames[-11:]
        tot_wts = 0
        wtd_frame = np.zeros_like(candidates[0])
        for j, candidate in enumerate(candidates):
            wt = j
            wtd_frame += wt * candidate
            tot_wts += wt
        wtd_frame = wtd_frame / tot_wts
        wtd_frame[wtd_frame < 1.0] = 0

        self._wtd_frames.append(wtd_frame)

        labels = label(wtd_frame)
        boxes = []
        for carnum in range(1, labels[1] + 1):
            nonzero = (labels[0] == carnum).nonzero()
            nonzero_y = np.array(nonzero[0])
            nonzero_x = np.array(nonzero[1])
            box = Box(
                top_left=Point(x=np.min(nonzero_x), y=np.min(nonzero_y)),
                bottom_right=Point(x=np.max(nonzero_x), y=np.max(nonzero_y))
            )
            boxes.append(box)

        return boxes
