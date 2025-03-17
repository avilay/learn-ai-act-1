import os.path as path
import os
from glob import glob
import pickle

from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import draw

DATAROOT = '/data/comp-vision/camera-calibration'


def main():
    num_row_points = 8
    num_col_points = 6
    objp = np.zeros((num_col_points * num_row_points, 3), np.float32)
    i = 0
    for y in range(num_col_points):
        for x in range(num_row_points):
            objp[i] = [x, y, 0]
            i += 1

    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    img_paths = glob(path.join(DATAROOT, 'GOPR*.jpg'))
    for img_path in img_paths:
        img = imread(img_path)
        name = path.basename(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (num_row_points, num_col_points), None)
        if ret:
            img2 = cv2.drawChessboardCorners(img, (num_row_points, num_col_points), corners, ret)
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f'Unable to find chessboard corners in {name}')

    test_img = imread(path.join(DATAROOT, 'test_image.jpg'))
    test_img_size = (test_img.shape[1], test_img.shape[0])
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, test_img_size, None, None)
    fixed_img = cv2.undistort(test_img, mtx, dist, None, mtx)

    calibrate_params = {
        'mtx': mtx,
        'dist': dist
    }
    pklfile = path.join(DATAROOT, 'calibrate_params.pkl')
    with open(pklfile, 'wb') as f:
        pickle.dump(calibrate_params, f)

    draw(
        1, 2,
        ('Original Image', test_img),
        ('Fixed Image', fixed_img)
    )


if __name__ == '__main__':
    main()
