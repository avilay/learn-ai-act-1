import os.path as path
import cv2
import numpy as np
from scipy.misc import imread, imsave
from moviepy.editor import VideoFileClip
import shortuuid
from windowing import *

DATAROOT = '/data/vehicle-tracking'


def edgify(img):
    ch2 = img[:, :, 1]
    sobx = cv2.Sobel(ch2, cv2.CV_64F, 1, 0)
    soby = cv2.Sobel(ch2, cv2.CV_64F, 0, 1)
    abs_sob = np.sqrt(np.square(sobx) + np.square(soby))
    scaled = np.uint8(255 * abs_sob / np.max(abs_sob))
    edges = np.zeros_like(scaled)
    edges[scaled > 180] = 1
    return edges


def snip_lanes(img, x_start, x_stop, y_start, y_stop):
    boxes = get_windows(img, x_start=x_start, x_stop=x_stop, y_start=y_start, y_stop=y_stop, win_size=64, stride=8)
    snips = []
    dbg_snips = []
    for i, box in enumerate(boxes, start=1):
        snip = img[box.top_left.y:box.bottom_right.y, box.top_left.x:box.bottom_right.x, :]
        if snip.shape[0] == 64 and snip.shape[1] == 64:
            edged = edgify(snip)
            if np.sum(edged) > 25:
                snips.append(snip)
            else:
                dbg_snips.append(snip)
    return dbg_snips, snips


def snip_roads(img, x_start, x_stop, y_start, y_stop):
    boxes = get_windows(img, x_start=x_start, x_stop=x_stop, y_start=y_start, y_stop=y_stop, win_size=64, stride=16)
    snips = []
    for box in boxes:
        snip = img[box.top_left.y:box.bottom_right.y, box.top_left.x:box.bottom_right.x, :]
        if snip.shape[0] == 64 and snip.shape[1] == 64:
            snips.append(snip)
    return snips


def main():
    frames = [
        # imread(path.join(DATAROOT, 'extracted_images2', 'test_735.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_736.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_750.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_811.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_815.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_816.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_840.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_841.jpg')),
        # imread(path.join(DATAROOT, 'extracted_images2', 'test_864.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_874.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_880.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_885.jpg')),
        imread(path.join(DATAROOT, 'extracted_images2', 'test_915.jpg')),
    ]
    for frame in frames:
        roads = snip_roads(frame, x_start=460, x_stop=675, y_start=415, y_stop=540)
        for j, road in enumerate(roads, start=1):
            uid = shortuuid.uuid()[:4]
            outfile = path.join(DATAROOT, 'non-vehicles', 'DbgExtraRoads', 'other2', f'snip_{uid}.jpg')
            imsave(outfile, road)


if __name__ == '__main__':
    main()

