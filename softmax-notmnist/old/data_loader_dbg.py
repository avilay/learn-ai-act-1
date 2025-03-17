import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile
import math
import matplotlib.image as mpimg
import os.path as path


def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('/'):
                with zipf.open(filename) as image_file:
                    image = Image.open(image_file)
                    image.load()
                    # Load image data as 1 dimensional array
                    # We're using float32 to save on memory space
                    feature = np.array(image, dtype=np.float32).flatten()

                # Get the the letter from the filename.  This is the letter of the image.
                label = os.path.split(filename)[1][0]

                features.append(feature)
                labels.append(label)
    return np.array(features), np.array(labels)


def load_imgs(filepath):
    features = []
    labels = []
    filenames_pbar = tqdm(os.listdir(filepath), unit='files')
    for filename in filenames_pbar:
        if filename.endswith('.png'):
            imgpath = path.join(filepath, filename)
            img = mpimg.imread(imgpath)
            feature = np.array(img, dtype=np.float32).flatten()
            label = path.basename(imgpath)[0]
            features.append(feature)
            labels.append(label)

    return np.array(features), np.array(labels)


