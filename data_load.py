from os.path import join

import numpy as np
import cv2
import settings

from os import listdir

from sklearn.utils import shuffle
from tqdm import tqdm


def slice_center(img, size):
    w, h = img.shape
    origin = (w - size) // 2
    return img[origin:origin + size, origin:origin + size]


def load_data(path):
    files = listdir(path)
    files = shuffle(files)

    x = []
    y = []
    for file in tqdm(files):
        img_x = cv2.imread(join(path, file), cv2.IMREAD_GRAYSCALE)
        img_x = img_x.astype(dtype=np.float32)
        img_x /= 255.
        # ground truth center slices
        img_y = slice_center(np.copy(img_x), size=21)

        w, h = img_x.shape
        img_x = cv2.GaussianBlur(img_x, (3, 3), 0)
        img_x = cv2.resize(img_x, dsize=(w // settings.SCALE_FACTOR, h // settings.SCALE_FACTOR))
        w, h = img_x.shape
        img_x = cv2.resize(img_x, dsize=(w * settings.SCALE_FACTOR, h * settings.SCALE_FACTOR), interpolation=cv2.INTER_CUBIC)

        # cv2.imshow('x', img_x)
        # cv2.imshow('y', img_y)
        # cv2.waitKey()

        img_x = np.expand_dims(img_x, axis=-1)
        img_y = np.expand_dims(img_y, axis=-1)

        x.append(img_x)
        y.append(img_y)

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y
