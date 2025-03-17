import numpy as np
from scipy.misc import imread
import cv2

from utils import draw


def main():
    imgpath = '/data/comp-vision/test4.jpg'
    img = imread(imgpath)
    # draw(1, 1, 'color', ('Original Image', img))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hue = hls[:, :, 0]
    light = hls[:, :, 1]
    sat = hls[:, :, 2]
    draw(
        2, 4,
        ('Original Image', img),
        ('Gray', gray),
        ('Red', red),
        ('Green', green),
        ('Blue', blue),
        ('Hue', hue),
        ('Light', light),
        ('Saturation', sat)
    )

    # Binarize gray, red, and saturation
    bin_gray = np.zeros_like(gray)
    bin_gray[(gray >= 180) & (gray <= 255)] = 1

    bin_red = np.zeros_like(red)
    bin_red[(red >= 200) & (red <= 255)] = 1

    print(np.min(sat), np.max(sat))
    bin_sat = np.zeros_like(sat)
    bin_sat[(sat >= 90) & (sat <= 255)] = 1

    draw(
        3, 2,
        ('Gray', gray),
        ('Bin Gray', bin_gray),
        ('Red', red),
        ('Bin Red', bin_red),
        ('Sat', sat),
        ('Bin Sat', bin_sat)
    )


if __name__ == '__main__':
    main()
