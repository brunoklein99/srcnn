from os.path import join, splitext

import cv2
import random

from os import listdir


def create_subimages(basename, savepath, fsub, stride):
    img = cv2.imread(basename, cv2.IMREAD_GRAYSCALE)
    w, h = img.shape
    for x in range(0, w - fsub + 1, stride):
        for y in range(0, h - fsub + 1, stride):
            sub = img[x:x+fsub, y:y+fsub]
            filename = '{}_{}-{}.bmp'.format(splitext(savepath)[0], x, y)
            cv2.imwrite(filename, sub)
            # cv2.imshow('', sub)
            # cv2.waitKey()


def create_dataset(path, fsub, stride):
    originals = listdir(path)
    for i in originals:
        create_subimages(join(path, i), join(path, '..', 'sub', i), fsub, stride)


def main():
    create_dataset('../data/91sr/original', fsub=33, stride=14)


if __name__ == "__main__":
    main()
