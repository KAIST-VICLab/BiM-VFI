import os

from datasets import register
from .data_utils import *

import torch
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image
from torch.utils.data import Dataset


@register('vimeo_septuplet')
class VimeoSeptuplet(Dataset):
    def __init__(self, root_path, patch_size=(224, 224), split='train', dis=False, **kwargs):
        super(VimeoSeptuplet, self).__init__()
        self.data_root = root_path
        self.mode = split
        self.patch_size = patch_size
        self.dis = dis
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, "r") as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, "r") as f:
            self.testlist = f.read().splitlines()
        cnt = int(len(self.trainlist) * 0.95)
        if self.mode == "train":
            self.img_list = self.trainlist[:cnt]
        elif self.mode == "test":
            self.img_list = self.testlist
        else:
            self.img_list = self.trainlist[cnt:]

    def get_img(self, index):

        if self.mode == "test":
            img_path = os.path.join(self.data_root, "sequences", self.img_list[index // 5])
            img_paths = [os.path.join(img_path, 'im{}.png'.format(i)) for i in range(1, 8)]
            img0 = read_image(img_paths[0])
            img1 = read_image(img_paths[6])
            imgt = read_image(img_paths[(index % 5) + 1])
            return img0, imgt, img1, ((index % 5) + 1) / 6.0, [1, (index % 5) + 2, 7]
        else:
            img_path = os.path.join(self.data_root, "sequences", self.img_list[index])
            img_paths = [os.path.join(img_path, 'im{}.png'.format(i)) for i in range(1, 8)]
            ind = list(range(7))
            random.shuffle(ind)
            ind = ind[:3]
            ind.sort()
            img0 = read_image(img_paths[ind[0]])
            imgt = read_image(img_paths[ind[1]])
            img1 = read_image(img_paths[ind[2]])
            return img0, imgt, img1, (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6), []

    def __getitem__(self, item):
        img0, imgt, img1, embt, scene_names = self.get_img(item)
        time_step = torch.Tensor([embt]).reshape(1, 1, 1)
        if self.mode == "train":
            img0, imgt, img1 = random_crop(img0, imgt, img1, self.patch_size)
            if random.random() > 0.5:
                img0, imgt, img1 = random_hor_flip(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1 = random_ver_flip(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1 = random_color_permutation(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1, time_step = random_temporal_flip(img0, imgt, img1, time_step)
            degree = random.randint(0, 3)
            img0, imgt, img1 = random_rotation(img0, imgt, img1, degree)
            scene_names = [os.path.join(self.img_list[item // 5], 'im1.png'), os.path.join(self.img_list[item // 5], f'im{(item % 5) + 2}.png'), os.path.join(self.img_list[item // 5], 'im7.png')]
            input_dict = {
                'img0': TF.to_dtype(img0, torch.float32, scale=True), 'imgt': TF.to_dtype(imgt, torch.float32, scale=True), 'img1': TF.to_dtype(img1, torch.float32, scale=True), 'time_step': time_step, 'scene_names': scene_names,
            }
        else:
            scene_names = [os.path.join(self.img_list[item // 5], 'im1.png'), os.path.join(self.img_list[item // 5], f'im{(item % 5) + 2}.png'), os.path.join(self.img_list[item // 5], 'im7.png')]
            input_dict = {
                'img0': TF.to_dtype(img0, torch.float32, scale=True), 'imgt': TF.to_dtype(imgt, torch.float32, scale=True), 'img1': TF.to_dtype(img1, torch.float32, scale=True), 'time_step': time_step, 'scene_names': scene_names
            }

        return input_dict

    def __len__(self):
        if self.mode == 'train':
            return len(self.img_list)
        else:
            return len(self.img_list) * 5
