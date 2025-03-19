import os

from datasets import register
from .data_utils import *

import torch
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image
from torch.utils.data import Dataset


@register('vimeo')
class Vimeo(Dataset):
    def __init__(self, root_path, patch_size=(224, 224), split='train', dis=False, **kwargs):
        super(Vimeo, self).__init__()
        self.data_root = root_path
        self.mode = split
        self.patch_size = patch_size
        self.dis = dis
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
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
        img_path = os.path.join(self.data_root, "sequences", self.img_list[index])
        img0 = read_image(os.path.join(img_path, "im1.png"))
        imgt = read_image(os.path.join(img_path, "im2.png"))
        img1 = read_image(os.path.join(img_path, "im3.png"))
        if self.dis:
            dis0 = torch.tensor(np.load(os.path.join(img_path, "dis_index_0_1_2.npy"))[None, :, :]).to(torch.float32)
            dis1 = torch.tensor(np.load(os.path.join(img_path, "dis_index_2_1_0.npy"))[None, :, :]).to(torch.float32)
        else:
            dis0 = None
            dis1 = None
        return img0, imgt, img1, [os.path.join(self.img_list[index], "im1.png"), os.path.join(self.img_list[index], "im2.png"), os.path.join(self.img_list[index], "im3.png")]

    def __getitem__(self, item):
        img0, imgt, img1, scene_names = self.get_img(item)
        time_step = torch.Tensor([0.5]).reshape(1, 1, 1)
        if self.mode == "train":
            if random.random() > 0.9:
                img0, imgt, img1 = random_resize(img0, imgt, img1)
            img0, imgt, img1 = random_crop(img0, imgt, img1, self.patch_size)
            if random.random() > 0.5:
                img0, imgt, img1 = random_hor_flip(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1, flowt0, flowt1, dis0, dis1 = random_ver_flip(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1 = random_color_permutation(img0, imgt, img1)
            if random.random() > 0.5:
                img0, imgt, img1, time_step = random_temporal_flip(img0, imgt, img1, time_step)
            degree = random.randint(0, 3)
            img0, imgt, img1 = random_rotation(img0, imgt, img1, degree)
        input_dict = {
            'img0': TF.to_dtype(img0, torch.float32, scale=True), 'imgt': TF.to_dtype(imgt, torch.float32, scale=True), 'img1': TF.to_dtype(img1, torch.float32, scale=True), 'time_step': time_step, 'scene_names': scene_names
        }
        return input_dict

    def __len__(self):
        return len(self.img_list)
