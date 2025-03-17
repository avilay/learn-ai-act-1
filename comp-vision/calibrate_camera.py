from scipy.misc import imread
import cv2
import os.path as path
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def main():
    img_path = '/data/camera-calibration/GOPR0032.jpg'
    img = imread(img_path)
    plt.imshow(img)
    plt.show()
    
    num_row_points = 8
    num_col_points = 6

    objp = np.zeros((num_col_points*num_row_points, 3), np.float32)
    i = 0
    for y in range(num_col_points):
        for x in range(num_row_points):
            objp[i] = [x, y, 0]
            i += 1

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.show()

    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    img2 = cv2.drawChessboardCorners(img, (8, 6), corners, ret)
    plt.imshow(img2)
    plt.show()
    

if __name__ == '__main__':
    main()
