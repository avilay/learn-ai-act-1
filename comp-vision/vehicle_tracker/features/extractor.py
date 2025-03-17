from functools import partial
from multiprocessing import Pool

import numpy as np
import cv2
from skimage.feature import hog

from .box import Box, Point
from .feature_spec import Channel, IMG_SIZE, COLOR_SPACE


def get_windows(frame, x_start=None, x_stop=None, y_start=None, y_stop=None, win_size=64, stride=32):
    """
    Returns a list of Box objects in a grid denoting windows on the frame
    """
    windows = []
    x_start = x_start if x_start else 0
    x_stop = x_stop if x_stop else frame.shape[1]
    y_start = y_start if y_start else 0
    y_stop = y_stop if y_stop else frame.shape[0]
    frame = frame[y_start:y_stop, x_start:x_stop]

    xlen, ylen = frame.shape[1], frame.shape[0]
    for y in range(0, ylen, stride):
        for x in range(0, xlen, stride):
            box = Box(
                top_left=Point(x=x+x_start, y=y+y_start),
                bottom_right=Point(x=x+x_start+win_size, y=y+y_start+win_size)
            )
            if box.bottom_right.x < x_stop and box.bottom_right.y < y_stop:
                windows.append(box)
    return windows


def bin_spatial(img, spec):
    features = cv2.resize(img, (spec.size, spec.size)).ravel()
    return features


def color_hist(img, spec):
    ch1 = img[:, :, 0]
    ch2 = img[:, :, 1]
    ch3 = img[:, :, 2]
    bins = spec.bins

    if spec.channel == Channel.FIRST:
        return np.histogram(ch1, bins)[0]
    elif spec.channel == Channel.SECOND:
        return np.histogram(ch2, bins)[0]
    elif spec.channel == Channel.THIRD:
        return np.histogram(ch3, bins)[0]
    elif spec.channel == Channel.ALL:
        ch1hist = np.histogram(ch1, bins, range=(0, 256))[0]
        ch2hist = np.histogram(ch2, bins, range=(0, 256))[0]
        ch3hist = np.histogram(ch3, bins, range=(0, 256))[0]
        return np.concatenate((ch1hist, ch2hist, ch3hist))


def hogs(img, spec):
    get_hog = partial(
        hog,
        pixels_per_cell=(spec.pixels_per_cell, spec.pixels_per_cell),
        cells_per_block=(spec.cells_per_block, spec.cells_per_block),
        transform_sqrt=True
    )
    if spec.channel == Channel.FIRST:
        return get_hog(img[:, :, 0])
    elif spec.channel == Channel.SECOND:
        return get_hog(img[:, :, 1])
    elif spec.channel == Channel.THIRD:
        return get_hog(img[:, :, 2])
    elif spec.channel == Channel.ALL:
        ch1hog = get_hog(img[:, :, 0])
        ch2hog = get_hog(img[:, :, 1])
        ch3hog = get_hog(img[:, :, 2])
        return np.concatenate((ch1hog, ch2hog, ch3hog))


def extract_img_features(img, feature_spec):
    if feature_spec.color_space != 'RGB':
        img = cv2.cvtColor(img, COLOR_SPACE[feature_spec.color_space])
    xvec = []
    if feature_spec.spatial_spec:
        spatial_features = bin_spatial(img, feature_spec.spatial_spec)
        xvec.append(spatial_features)
    if feature_spec.hist_spec:
        hist_features = color_hist(img, feature_spec.hist_spec)
        xvec.append(hist_features)
    if feature_spec.hog_spec:
        hog_features = hogs(img, feature_spec.hog_spec)
        xvec.append(hog_features)
    xvec = np.concatenate(xvec)
    return xvec


def extract_frame_features(frame, windows, feature_spec):
    extract_img_features_with_spec = partial(extract_img_features, feature_spec=feature_spec)

    win_imgs = []
    for window in windows:
        win_img = frame[window.top_left.y:window.bottom_right.y, window.top_left.x:window.bottom_right.x]
        win_img = cv2.resize(win_img, (IMG_SIZE, IMG_SIZE))
        win_imgs.append(win_img)

    with Pool() as pool:
        results = pool.map_async(extract_img_features_with_spec, win_imgs)
        xvecs = results.get()

    return xvecs