import os.path as path
import pickle
from glob import glob

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

from ..features.feature_spec import HogSpec, Channel, HistogramSpec, FeatureSpec, SpatialSpec
from ..features.extractor import get_windows, extract_frame_features
from .frame_cache import FrameCache

DATAROOT = '/data/vehicle-tracking'


def to_image(frame):
    # Scale the heat values to 0-255
    numerator = 255 * (frame - np.min(frame))
    denominator = np.max(frame) - np.min(frame)
    img_frame = numerator // denominator
    img_frame = img_frame.astype(np.uint8)
    return img_frame


def plot(frame_ndx, nrows, ncols, *imgs):
    if nrows * ncols != len(imgs):
        raise RuntimeError(f'{nrows} x {ncols} != {len(imgs)}')

    plt.figure(figsize=(24, 9))
    for i, (title, img) in enumerate(imgs, start=1):
        fig = plt.subplot(nrows, ncols, i)
        fig.imshow(img, cmap='hot')
        fig.set_title(title)

    outfile = path.join(DATAROOT, 'heatmaps', f'frame{frame_ndx}.png')
    plt.savefig(outfile)
    plt.close()


def create_findcars(model, spec, frame_cache=None):
    scaler = model['scaler']
    svm = model['svm']
    ver = model['ver']
    print(f'Using model version {ver}')

    def find_cars(frame):
        hot_wins = []

        win64 = get_windows(frame, x_start=400, y_start=400, y_stop=660, win_size=64, stride=16)
        win112 = get_windows(frame, x_start=400, y_start=400, y_stop=660, win_size=112, stride=16)
        wins = win64 + win112

        X = np.array(extract_frame_features(frame, wins, spec))
        X = scaler.transform(X)

        likelihoods = svm.decision_function(X)
        for win, likelihood in zip(wins, likelihoods):
            if likelihood > 0.9:
                hot_wins.append(win)

        if frame_cache:
            frame_cache.add(frame, hot_wins)
            boxes = frame_cache.car_boxes()
        else:
            boxes = hot_wins

        # Draw hot windows on the input frame
        for box in boxes:
            cv2.rectangle(frame, box.top_left, box.bottom_right, (0, 255, 0), thickness=2)

        return frame

    return find_cars


def rehydrate():
    with open(path.join(DATAROOT, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model


def image_pipeline():
    frames_dir = path.join(DATAROOT, 'extracted_images')
    outdir = path.join(DATAROOT, 'out_images')
    frames = []
    num_files = len(glob(path.join(frames_dir, '*.jpg')))
    for i in range(1, num_files+1):
        imgfile = path.join(frames_dir, f'frame{i}.jpg')
        frames.append(imread(imgfile))

    model = rehydrate()

    hog_spec = HogSpec(
        channel=Channel.ALL,
        orientations=11,
        pixels_per_cell=16,
        cells_per_block=2
    )
    hist_spec = HistogramSpec(channel=Channel.SECOND, bins=32)
    spatial_spec = SpatialSpec(size=32)
    spec = FeatureSpec(color_space='YCrCb', hog_spec=hog_spec, hist_spec=hist_spec, spatial_spec=spatial_spec)

    # find_cars = create_findcars(model, spec)
    fc = FrameCache()
    find_cars = create_findcars(model, spec, fc)
    for i, frame in enumerate(frames):
        print(f'Processing frame {i}')
        annotated_frame = find_cars(frame)
        imgfile = path.join(outdir, f'frame{i}.jpg')
        imsave(imgfile, annotated_frame)

        heatmaps, wtd_heatmap = fc.heat_maps()
        if heatmaps:
            imgs = []
            for h, heatmap in enumerate(heatmaps):
                imgs.append((f'frame{h}', to_image(heatmap)))
            imgs.append(('wtd frame', to_image(wtd_heatmap)))
            plot(i, 3, 4, *imgs)


def video_pipeline():
    in_file = path.join(DATAROOT, 'in_videos', 'cars.mp4')
    out_file = path.join(DATAROOT, 'out_videos', 'cars_1.mp4')

    model = rehydrate()

    hog_spec = HogSpec(
        channel=Channel.ALL,
        orientations=11,
        pixels_per_cell=16,
        cells_per_block=2
    )
    hist_spec = HistogramSpec(channel=Channel.SECOND, bins=32)
    spatial_spec = SpatialSpec(size=32)
    spec = FeatureSpec(color_space='YCrCb', hog_spec=hog_spec, hist_spec=hist_spec, spatial_spec=spatial_spec)

    find_cars = create_findcars(model, spec)

    video = VideoFileClip(in_file)
    out_video = video.fl_image(find_cars)
    out_video.write_videofile(out_file, audio=False)


if __name__ == '__main__':
    # video_pipeline()
    image_pipeline()
