import os.path as path
import sys

import numpy as np
import cv2
from scipy.misc import imread

from utils import draw

DATAROOT = '/data/comp-vision'


def gradient(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return scaled_sobelx


def binarize(img, bounds):
    bin_img = np.zeros_like(img)
    bin_img[(img >= bounds[0]) & (img <= bounds[1])] = 1
    return bin_img


def stack(shape, reds=None, blues=None, greens=None):
    if reds is None and blues is None and greens is None:
        raise RuntimeError('All channels cannot be None!')
    if reds is None:
        reds = np.full(shape, fill_value=0.25)
    if blues is None:
        blues = np.full(shape, fill_value=0.25)
    if greens is None:
        greens = np.full(shape, fill_value=0.25)
    stacked = np.dstack((reds, blues, greens))
    stacked = stacked.astype(np.float32)
    return stacked


def combine(shape, img1=None, img2=None, img3=None):
    if img1 is None and img2 is None and img3 is None:
        raise RuntimeError('All images cannot be None!')
    if img1 is None:
        img1 = np.zeros(shape)
    if img2 is None:
        img2 = np.zeros(shape)
    if img3 is None:
        img3 = np.zeros(shape)
    combo = np.zeros(shape)
    combo[(img1 == 1) | (img2 == 1) | (img3 == 1)] = 1
    combo = combo.astype(np.float32)
    return combo


def main():
    imgfile = sys.argv[1]
    if len(sys.argv) >= 4:
        grad_name = sys.argv[2]
        bin_name = sys.argv[3]
        show_comps = False
    else:
        show_comps = True

    img = imread(imgfile)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    components = {
        'reds': img[:, :, 0],
        'blues': img[:, :, 1],
        'greens': img[:, :, 2],
        'gray': cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
        'hues': hls[:, :, 0],
        'lights': hls[:, :, 1],
        'saturations': hls[:, :, 2]
    }

    if show_comps:
        draw(
            2, 4,
            ('Original Image', img),
            ('Reds', components['reds']),
            ('Blues', components['blues']),
            ('Greens', components['greens']),
            ('Gray', components['gray']),
            ('Hues', components['hues']),
            ('Lights', components['lights']),
            ('Saturations', components['saturations'])
        )
        exit(0)

    grad_img = components[grad_name]
    bin_img = components[bin_name]

    grad_bounds = [
        (20, 100),
        (20, 120),
        (20, 150)
    ]

    bin_bounds = [
        (130, 255),
        (140, 255),
        (170, 255)
    ]

    for grad_bound in grad_bounds:
        for bin_bound in bin_bounds:
            grad_bin = binarize(gradient(components[grad_name]), grad_bound)
            only_bin = binarize(components[bin_name], bin_bound)
            stacked = stack(img.shape[:2], blues=grad_bin, greens=only_bin)
            combo = combine(img.shape[:2], img2=grad_bin, img3=only_bin)

            # draw(
            #     2, 3,
            #     ('Original Image', img),
            #     (f'Grad {grad_name}', grad_bin),
            #     (f'Color Channel {bin_name}', only_bin),
            #     ('Stacked', stacked),
            #     ('Combined', combo),
            #     ('Original Image', img)
            # )
            #

            draw(
                2, 2,
                ('Original', img),
                (f'Grad {grad_name} {grad_bound}', grad_bin),
                (f'Bin {bin_name} {bin_bound}', only_bin),
                ('Combined', combo)
            )


if __name__ == '__main__':
    main()
