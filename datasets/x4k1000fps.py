import numpy as np
import cv2
from glob import glob
import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .datasets import register


@register('xtest')
class X_Test(Dataset):
    def __init__(self, root_path, split, **kwargs):
        self.test_data_path = root_path
        if split == 'multiple':
            self.multiple = 4
        else:
            self.multiple = 16
        self.testPath = self.make_2_d_dataset_x_test(
            self.test_data_path, self.multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " \
                                + self.test_data_path + "\n"))

    def make_2_d_dataset_x_test(self, test_data_path, multiple, t_step_size):
        """ make [I0,I1,It,t,scene_folder] """
        """ 1D (accumulated) """
        testPath = []
        for type_folder in sorted(glob(os.path.join(test_data_path, '*', ''))):  # [type1,type2,type3,...]
            for scene_folder in sorted(glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
                frame_folder = sorted(glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
                for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                    if idx == len(frame_folder) - 1:
                        break
                    for mul in range(multiple, t_step_size, multiple):
                        I0I1It_paths = []
                        I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                        I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                        I0I1It_paths.append(frame_folder[idx + mul])  # It
                        I0I1It_paths.append(mul / t_step_size)
                        I0I1It_paths.append([frame_folder[idx].replace(test_data_path + '/', ''), frame_folder[idx + mul].replace(test_data_path + '/', ''), frame_folder[idx + t_step_size].replace(test_data_path + '/', '')])  # type1/scene1
                        testPath.append(I0I1It_paths)
        return testPath

    def frames_loader_test(self, I0I1It_Path):
        frames = []
        for path in I0I1It_Path:
            frame = cv2.imread(path)[:, :, ::-1]
            frames.append(frame)

        return frames

    def RGBframes_np2Tensor(self, imgIn, channel=3):
        ## input : T, H, W, C
        if channel == 1:
            # rgb --> Y (gray)
            imgIn = np.sum(
                imgIn * np.reshape(
                    [65.481, 128.553, 24.966], [1, 1, 1, 3]
                ) / 255.0,
                axis=3,
                keepdims=True) + 16.0

        # to Tensor
        ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

        return imgIn

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_names = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]
        frames = self.frames_loader_test(I0I1It_Path)
        img0 = TF.to_tensor(frames[0].copy())
        imgt = TF.to_tensor(frames[2].copy())
        img1 = TF.to_tensor(frames[1].copy())
        time_step = torch.Tensor([t_value]).reshape(1, 1, 1)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return {
            'img0': img0, 'imgt': imgt, 'img1': img1, 'time_step': time_step, 'scene_names': scene_names
        }

    def __len__(self):
        return self.nIterations
