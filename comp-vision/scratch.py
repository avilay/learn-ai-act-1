import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.misc import imread
import os.path as path
import sys


def scatter(points, labels):
    points = points.reshape((-1, 3))
    colors = points / 255

    xvals = points[:, 0]
    yvals = points[:, 1]
    zvals = points[:, 2]

    xlbl = labels[0]
    ylbl = labels[1]
    zlbl = labels[2]

    plt.figure()
    xy = plt.subplot(221)
    xy.scatter(xvals, yvals, c=colors)
    xy.set_xlabel(xlbl)
    xy.set_ylabel(ylbl)

    yz = plt.subplot(222)
    yz.scatter(yvals, zvals, c=colors)
    yz.set_xlabel(ylbl)
    yz.set_ylabel(zlbl)

    xz = plt.subplot(223)
    xz.scatter(xvals, zvals, c=colors)
    xz.set_xlabel(xlbl)
    xz.set_ylabel(zlbl)

    fig = plt.subplot(224, projection='3d')
    fig.scatter(xvals, yvals, zvals, c=colors)
    fig.set_xlabel(xlbl)
    fig.set_ylabel(ylbl)
    fig.set_zlabel(zlbl)


def main():
    imgfile = path.join('/data/comp-vision/color-spaces', sys.argv[1])
    img = imread(imgfile)

    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                           interpolation=cv2.INTER_NEAREST)

    # Convert subsampled image to desired color space(s)
    rgb = img_small
    hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(img_small, cv2.COLOR_RGB2HLS)
    lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2Lab)
    luv = cv2.cvtColor(img_small, cv2.COLOR_RGB2Luv)

    scatter(rgb, 'RGB')
    scatter(hsv, 'HSV')
    scatter(hls, 'HLS')
    scatter(lab, 'LAB')
    scatter(luv, 'LUV')

    plt.show()

if __name__ == '__main__':
    main()
