#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image


def __fix_size(orig_size, needed_size):
    """calculate size keeping the aspect ratio"""
    # 1. find small side
    smaller_side_index = np.argmin(orig_size)
    bigger_side_index = 1 - smaller_side_index
    # 2. scale by small side
    aspect_ratio = orig_size[smaller_side_index] / needed_size[smaller_side_index]
    # 3. return new size with correct aspect ratio
    size = [0, 0]
    size[smaller_side_index] = needed_size[smaller_side_index]
    size[bigger_side_index] = int(needed_size[bigger_side_index] / aspect_ratio)
    return size


def __square(size):
    return size[0] == size[1]


def __roi(size_with_ratio, out_size):
    """calculate region of interest"""
    # 1. find small side
    smaller_side_index = np.argmin(size_with_ratio)
    bigger_side_index = 1 - smaller_side_index
    # 2. calculate region of interest
    #    a. small side is untouched
    #    b. bigger side is cropped
    box = [0, 0, 0, 0]
    # 2.a
    box[smaller_side_index] = 0
    box[smaller_side_index+2] = out_size[smaller_side_index]
    # 2.b
    bigger_side_center = int(size_with_ratio[bigger_side_index] / 2)
    box[bigger_side_index] = bigger_side_center - out_size[bigger_side_index] / 2
    # + whole number to eliminate problems with non-integer centers:
    box[bigger_side_index+2] = box[bigger_side_index] + out_size[bigger_side_index]
    return box


def preprocess_images(image_dir, out_size):
    """preprocess images"""
    def path_join(left, right):
        return os.path.join(left, right)

    if not __square(out_size):
        raise NotImplementedError

    image_files = os.listdir(image_dir)
    good_images = []
    bad_files = []
    # 1. load image - if fails, skip
    for file in image_files:
        fp = open(path_join(image_dir, file), "rb")
        try:
            img = Image.open(fp).convert('RGB')
            size_with_ratio = __fix_size(img.size, out_size)
            img = img.resize(size_with_ratio, Image.HAMMING)
            if not __square(size_with_ratio):  # crop center if image is not a square
                img = img.crop(box=__roi(size_with_ratio, out_size))
            # 2. create tuple of image name and resized image object
            good_images.append((file, img))
        except IOError:  # bad image/not an image
            bad_files.append(file)
    # 3. return tuples list and list of bad images (for debug primarily)
    return good_images, bad_files
