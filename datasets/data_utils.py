import cv2
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
import torch
import numpy as np
import random

perm = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
rotate = [90, 180, 270]


def random_crop(img0, imgt, img1, crop_size):
    im_h, im_w = img0.shape[-2:]
    crop_h, crop_w = crop_size, crop_size
    i = random.randint(0, im_h - crop_h)
    j = random.randint(0, im_w - crop_w)
    img0 = img0[:, i:i + crop_h, j:j + crop_w]
    imgt = imgt[:, i:i + crop_h, j:j + crop_w]
    img1 = img1[:, i:i + crop_h, j:j + crop_w]
    return img0, imgt, img1


def random_hor_flip(img0, imgt, img1):
    img0, imgt, img1 = TF.horizontal_flip(img0), TF.horizontal_flip(imgt), TF.horizontal_flip(img1)
    return img0, imgt, img1


def random_ver_flip(img0, imgt, img1):
    img0, imgt, img1 = TF.vertical_flip(img0), TF.vertical_flip(imgt), TF.vertical_flip(img1)
    return img0, imgt, img1


def random_color_permutation(img0, imgt, img1):
    perm_idx = random.randint(0, 5)
    img0, imgt, img1 = TF.permute_channels(img0, perm[perm_idx]), TF.permute_channels(imgt, perm[perm_idx]), TF.permute_channels(img1, perm[perm_idx])
    return img0, imgt, img1


def random_temporal_flip(img0, imgt, img1, time_step):
    tmp = img1
    img1 = img0
    img0 = tmp
    time_step = 1 - time_step
    return img0, imgt, img1, time_step


def random_rotation(img0, imgt, img1, degree):
    if degree != 3:
        img0 = TF.rotate(img0, rotate[degree])
        imgt = TF.rotate(imgt, rotate[degree])
        img1 = TF.rotate(img1, rotate[degree])
    return img0, imgt, img1


def random_resize(img0, imgt, img1):
    h, w = img0.shape[-2:]
    img0 = TF.resize(img0, [2*h, 2*w])
    imgt = TF.resize(imgt, [2*h, 2*w])
    img1 = TF.resize(img1, [2*h, 2*w])
    return img0, imgt, img1


def read_flow(name):
    with open(name, "rb") as f:
        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

