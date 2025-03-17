import numpy as np
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt

from utils import draw


def gradient(img, orient='x', kernel=3, bound=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dxdy = {
        'x': (1, 0),
        'y': (0, 1)
    }
    dx, dy = dxdy[orient]
    sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    sbin = np.zeros_like(scaled_sobel)
    sbin[(scaled_sobel >= bound[0]) & (scaled_sobel <= bound[1])] = 1
    return sbin


def gradient_magnitude(img, kernel=3, bound=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    mag = np.sqrt(np.square(sobelx), np.square(sobely))
    scaled_sobel = np.uint8(255 * mag / np.max(mag))
    sbin = np.zeros_like(scaled_sobel)
    sbin[(scaled_sobel >= bound[0]) & (scaled_sobel <= bound[1])] = 1
    return sbin


def gradient_direction(img, kernel=3, bound=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    angles = np.arctan2(abs_sobely, abs_sobelx)
    sbin = np.zeros_like(angles)
    sbin[(angles >= bound[0]) & (angles <= bound[1])] = 1
    return sbin


def main():
    img = imread('/data/comp-vision/signs_vehicles_xygrad.png')

    kernel = 3
    gradx = gradient(img, kernel=kernel, bound=(30, 100))
    grady = gradient(img, orient='y', kernel=kernel, bound=(30, 100))
    mag = gradient_magnitude(img, kernel=kernel, bound=(30, 100))
    dirgrad = gradient_direction(img, kernel=kernel, bound=(0.7, 1.3))
    draw(
        2, 2,
        ('Grad X', gradx),
        ('Grad Y', grady),
        ('Magnitude', mag),
        ('Grad Direction', dirgrad)
    )

    combined = np.zeros_like(dirgrad)
    combined[((gradx == 1) & (grady == 1)) | ((mag == 1) & (dirgrad == 1))] = 1
    draw(1, 1, ('Combined', combined))


if __name__ == '__main__':
    main()
