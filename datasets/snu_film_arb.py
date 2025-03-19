import os
import cv2

from datasets import register

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


@register('snu_film_arb')
class SNUFilmArb(Dataset):
    def __init__(self, root_path, split="extreme"):
        self.data_root = root_path
        self.data_type = split
        assert split in ["medium", "hard", "extreme"]
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.data_type == "medium":
            medium_file = os.path.join(self.data_root, "eval_modes/test-arb-medium.txt")
            with open(medium_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "hard":
            hard_file = os.path.join(self.data_root, "eval_modes/test-arb-hard.txt")
            with open(hard_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "extreme":
            extreme_file = os.path.join(self.data_root, "eval_modes/test-arb-extreme.txt")
            with open(extreme_file, 'r') as f:
                self.meta_data = f.read().splitlines()

    def get_img(self, index):
        imgpath = self.meta_data[index]
        imgpaths = imgpath.split()

        # Load images
        img0 = cv2.imread(os.path.join(self.data_root, '/'.join(imgpaths[0].split('/')[2:])))[:, :, ::-1]
        gt = cv2.imread(os.path.join(self.data_root, '/'.join(imgpaths[1].split('/')[2:])))[:, :, ::-1]
        img1 = cv2.imread(os.path.join(self.data_root, '/'.join(imgpaths[2].split('/')[2:])))[:, :, ::-1]
        time_step = (int(imgpaths[1].split('/')[-1].split('.')[0]) - int(imgpaths[0].split('/')[-1].split('.')[0])) / (int(imgpaths[2].split('/')[-1].split('.')[0]) - int(imgpaths[0].split('/')[-1].split('.')[0]))
        scene_name0 = '/'.join(imgpaths[0].split('/')[3:-1] + [imgpaths[0].split('/')[-1].replace('.png', ''), imgpaths[0].split('/')[-1]])
        scene_name1 = '/'.join(imgpaths[1].split('/')[3:-1] + [imgpaths[0].split('/')[-1].replace('.png', ''), imgpaths[1].split('/')[-1]])
        scene_name2 = '/'.join(imgpaths[2].split('/')[3:-1] + [imgpaths[0].split('/')[-1].replace('.png', ''), imgpaths[2].split('/')[-1]])

        return img0, gt, img1, [scene_name0, scene_name1, scene_name2], time_step

    def __getitem__(self, index):
        img0, imgt, img1, scene_names, time_step = self.get_img(index)
        img0 = TF.to_tensor(img0.copy())
        img1 = TF.to_tensor(img1.copy())
        imgt = TF.to_tensor(imgt.copy())
        time_step = torch.Tensor([time_step]).reshape(1, 1, 1)
        return {
            'img0': img0, 'imgt': imgt, 'img1': img1, 'time_step': time_step, 'scene_names': scene_names
        }
