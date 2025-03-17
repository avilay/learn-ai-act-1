import pickle
import os.path as path

import cv2
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

DATAROOT = '/data/comp-vision/camera-calibration'


def fix(distored_img, nx, ny, calibrate_params):
    mtx = calibrate_params['mtx']
    dist = calibrate_params['dist']
    undistorted_img = cv2.undistort(distored_img, mtx, dist, None, mtx)
    img_size = undistorted_img.shape[1], undistorted_img.shape[0]
    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        # In place mutation of undistorted_img
        cv2.drawChessboardCorners(undistorted_img, (nx, ny), corners, ret)
        src = np.array([corners[0][0], corners[7][0], corners[40][0], corners[47][0]], dtype=np.float32)
        dst = np.array([[100, 75], [1200, 75], [100, 900], [1200, 900]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        xformed_img = cv2.warpPerspective(undistorted_img, M, img_size, flags=cv2.INTER_LINEAR)
        return xformed_img
    else:
        print('Unable to find chessboard corners')
        return np.zeros(undistorted_img.shape)


def draw(orig_img, fixed_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(orig_img)
    ax1.set_title('Original Image')
    ax2.imshow(fixed_img)
    ax2.set_title('Fixed Image')
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.show()


def main():
    pklfile = path.join(DATAROOT, 'calibrate_params.pkl')
    with open(pklfile, 'rb') as f:
        calibrate_params = pickle.load(f)

    test_img = imread(path.join(DATAROOT, 'GOPR0060.jpg'))
    nx, ny = 8, 6
    fixed_img = fix(test_img, nx, ny, calibrate_params)
    draw(test_img, fixed_img)


if __name__ == '__main__':
    main()
