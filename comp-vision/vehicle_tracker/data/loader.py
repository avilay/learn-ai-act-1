"""
As a stand-alone script this module does any one of the following -
* Dedups images in GTI dirs and generates a text file with filenames of all unique images.
* Filters out road images from the non-vehicles/Extras dir.
* Split all learning data into train, validation, and test sets. It creates 3 text files containing
names of image filenames in each of the sets.

As a module called from another module it has iterators to load the training, validation, and test
data into image arrays one at a time.
"""
import numpy as np
import cv2
from scipy.misc import imread
from glob import glob
import os.path as path
from functools import partial
import shutil
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


DATAROOT = '/data/vehicle-tracking'
VEHICLES_DIR = path.join(DATAROOT, 'vehicles')
NON_VEHICLES_DIR = path.join(DATAROOT, 'non-vehicles')


hist = partial(np.histogram, bins=32, range=(0, 256))


def are_different(img1, img2):
    rhist1, _ = hist(img1[:, :, 0])
    ghist1, _ = hist(img1[:, :, 1])
    bhist1, _ = hist(img1[:, :, 2])

    rhist2, _ = hist(img2[:, :, 0])
    ghist2, _ = hist(img2[:, :, 1])
    bhist2, _ = hist(img2[:, :, 2])

    red_err = np.sqrt(mean_squared_error(rhist1, rhist2))
    green_err = np.sqrt(mean_squared_error(ghist1, ghist2))
    blue_err = np.sqrt(mean_squared_error(bhist1, bhist2))
    err = np.mean([red_err, green_err, blue_err])
    return err > 45


def are_close(img1, img2):
    return not are_different(img1, img2)


def dedup_gti():
    dirs_to_dedup = [
        (path.join(VEHICLES_DIR, 'GTI_Far'), path.join(VEHICLES_DIR, 'gti_far_unq.txt'), path.join(VEHICLES_DIR, 'gti_far_dup.txt')),
        (path.join(VEHICLES_DIR, 'GTI_Left'), path.join(VEHICLES_DIR, 'gti_left_unq.txt'), path.join(VEHICLES_DIR, 'gti_left_dup.txt')),
        (path.join(VEHICLES_DIR, 'GTI_MiddleClose'), path.join(VEHICLES_DIR, 'gti_mid_unq.txt'), path.join(VEHICLES_DIR, 'gti_mid_dup.txt')),
        (path.join(VEHICLES_DIR, 'GTI_Right'), path.join(VEHICLES_DIR, 'gti_right_unq.txt'), path.join(VEHICLES_DIR, 'gti_right_dup.txt')),
        (path.join(NON_VEHICLES_DIR, 'GTI'), path.join(NON_VEHICLES_DIR, 'gti_unq.txt'), path.join(NON_VEHICLES_DIR, 'gti_dup.txt'))
    ]
    for dir_to_dedup, unq_out, dup_out in dirs_to_dedup:
        uniq_files = []
        dup_files = []
        imgfiles = glob(path.join(dir_to_dedup, '*.png'))
        uniq_files.append(imgfiles[0])
        last_imgs = [imread(imgfiles[0])]
        for imgfile in imgfiles:
            img = imread(imgfile)
            match_found = False
            for last_img in last_imgs:
                if are_close(img, last_img):
                    match_found = True
                    break
            last_imgs.append(img)
            last_imgs = last_imgs[-5:]
            if not match_found:
                uniq_files.append(imgfile)
            else:
                dup_files.append(imgfile)

        with open(unq_out, 'wt') as f:
            for imgfile in uniq_files:
                print(imgfile, file=f)
        with open(dup_out, 'wt') as f:
            for imgfile in dup_files:
                print(imgfile, file=f)


def is_road(img):
    green_avg = np.mean(img[:, :, 1].ravel())
    blue_avg = np.mean(img[:, :, 2].ravel())
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hue_avg = np.mean(hls[:, :, 0].ravel())
    return green_avg < 100 and blue_avg < 100 and hue_avg > 85


def filter_roads():
    imgfiles = sorted(glob(path.join(NON_VEHICLES_DIR, 'Extras', '*.png')))
    roadimgs = []
    for imgfile in imgfiles:
        img = imread(imgfile)
        if is_road(img):
            roadimgs.append(imgfile)

    outfile = path.join(NON_VEHICLES_DIR, 'extras_roads.txt')
    with open(outfile, 'wt') as f:
        for roadimg in roadimgs:
            print(roadimg, file=f)

    outdir = path.join(NON_VEHICLES_DIR, 'ExtraRoads')
    for roadimg in tqdm(roadimgs):
        shutil.copy(roadimg, outdir)


def train_val_test_split():
    posfiles = [
        path.join(VEHICLES_DIR, 'gti_far_unq.txt'),
        path.join(VEHICLES_DIR, 'gti_left_unq.txt'),
        path.join(VEHICLES_DIR, 'gti_mid_unq.txt'),
        path.join(VEHICLES_DIR, 'gti_right_unq.txt')
    ]

    negfiles = [
        path.join(NON_VEHICLES_DIR, 'gti_unq.txt'),
    ]

    all = []

    # Add deduped GTI cars to negative samples
    for posfile in posfiles:
        with open(posfile, 'rt') as f:
            for line in f:
                line = line.strip()
                all.append((line, 'yes'))

    # Add KITTI cars to positive samples
    kitti_imgs = glob(path.join(VEHICLES_DIR, 'KITTI_Extracted', '*.png'))
    for kitti_img in kitti_imgs:
        all.append((kitti_img, 'yes'))

    # Add deduped GTI non-vehicles to negative samples
    for negfile in negfiles:
        with open(negfile, 'rt') as f:
            for line in f:
                line = line.strip()
                all.append((line, 'no'))

    # Add ExtraLanes to the negative samples
    # extralanes_imgs = glob(path.join(NON_VEHICLES_DIR, 'ExtraLanes', '*.jpg'))
    # for extralanes_img in extralanes_imgs:
    #     all.append((extralanes_img, 'no'))
    # print(f'Added {len(extralanes_imgs)} extra lane images')

    # Add ExtraRoads to the negative samples
    extraroads_imgs = glob(path.join(NON_VEHICLES_DIR, 'ExtraRoads', '*.png'))
    for extralanes_img in extraroads_imgs:
        all.append((extralanes_img, 'no'))
    print(f'Added {len(extraroads_imgs)} extra road images')

    np.random.shuffle(all)
    num_samples = len(all)
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.2)
    num_test = int(num_samples * 0.1)
    train = all[:num_train]
    val = all[num_train:num_train + num_val]
    test = all[num_train + num_val:]

    train_file = path.join(DATAROOT, 'train.txt')
    with open(train_file, 'wt') as f:
        for filename, label in train:
            print(f'{filename},{label}', file=f)

    val_file = path.join(DATAROOT, 'val.txt')
    with open(val_file, 'wt') as f:
        for filename, label in val:
            print(f'{filename},{label}', file=f)

    test_file = path.join(DATAROOT, 'test.txt')
    with open(test_file, 'wt') as f:
        for filename, label in test:
            print(f'{filename},{label}', file=f)


def train_data():
    train_filenames = []
    with open('/data/vehicle-tracking/train.txt', 'rt') as f:
        for line in f:
            line = line.strip()
            flds = line.split(',')
            train_filenames.append((flds[0], flds[1]))

    for filename, label in train_filenames:
        img = imread(filename)
        yield img, label


def val_data():
    val_filenames = []
    with open('/data/vehicle-tracking/val.txt', 'rt') as f:
        for line in f:
            line = line.strip()
            flds = line.split(',')
            val_filenames.append((flds[0], flds[1]))

    for filename, label in val_filenames:
        img = imread(filename)
        yield img, label


def main():
    # dedup_gti()
    # filter_roads()
    train_val_test_split()


if __name__ == '__main__':
    main()
