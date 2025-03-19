import glob
import numpy
import os
import cv2
import math
import PIL.Image
import torch
import torch.nn.functional as F
import tqdm
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
import sys

from torchvision.utils import save_image
from utils.flowvis import flow2img
from utils.padder import InputPadder


##########################################################

##########################################################
def inference_demo(model, ratio, video_path, out_path):
    videogen = []
    is_video = video_path.endswith(".mkv") or video_path.endswith(".webm") or video_path.endswith(
        ".mp4") or video_path.endswith(".avi")
    if is_video:
        clip = VideoFileClip(video_path)
        videogen = clip.iter_frames()
        ratio = 2
        fps = clip.fps
        # if fps == 23 or fps == 25:
        #     fps = 24
        # if fps == 29 or fps == 31:
        #     fps = 30
        # if fps == 59:
        #     fps = 60
        # ratio = 120 // fps
        # if fps == 60:
        #     ratio = 120 // 24
    else:
        for f in os.listdir(video_path):
            if 'jpg' or 'jpg' in f:
                videogen.append(f)
                videogen.sort(key=lambda x: int(x[:-4]))

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(out_path + "_flow"):
        os.mkdir(out_path + '_flow')

    img0 = None
    idx = 0
    name_idx = 0
    time_range = torch.arange(1, ratio).view(ratio - 1, 1, 1, 1).cuda() / ratio
    for curfile_name in videogen:
        if not is_video:
            curframe = os.path.join(video_path, curfile_name)
            img4_np = cv2.imread(curframe)[:, :, ::-1]
        else:
            img4_np = curfile_name
        img4 = (torch.tensor(img4_np.transpose(2, 0, 1).copy()).float() / 255.0).unsqueeze(0).cuda()
        if img0 is None:
            img0 = img4
            cv2.imwrite(out_path + '/{:0>7d}.jpg'.format(name_idx),
                        (img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1])
            _, _, h, w = img0.shape
            if h >= 2160:
                scale_factor = 0.25
                pyr_level = 7
                nr_lvl_skipped = 0
            elif h >= 1080:
                scale_factor = 0.5
                pyr_level = 6
                nr_lvl_skipped = 0
            else:
                scale_factor = 1
                pyr_level = 5
                nr_lvl_skipped = 0
            idx += 1
            name_idx += 1
            continue
        # if is_video:
        # if fps == 60:
        #     if idx % 5 != 0 and idx % 5 != 3:
        #         idx += 1
        #         continue
        # img0_ = F.interpolate(img0, scale_factor=pre_down, mode='bilinear')
        # img4_ = F.interpolate(img4, scale_factor=pre_down, mode='bilinear')
        for i in range(ratio - 1):
            dis0 = torch.ones((1, 1, h, w), device=img0.device) * (i / ratio)
            dis1 = 1 - dis0
            results_dict = model(img0=img0, img1=img4, time_step=time_range[i], dis0=dis0, dis1=dis1, scale_factor=scale_factor,
                                 ratio=(1 / scale_factor), pyr_level=pyr_level, nr_lvl_skipped=nr_lvl_skipped)
            imgt_pred = results_dict['imgt_pred']
            # imgt_preds = results_dict['imgt_preds']
            imgt_pred = torch.clip(imgt_pred, 0, 1)
            save_image(flow2img(results_dict['flowt0_pred_list'][0]),
                       os.path.join(out_path + '_flow', "{:0>7d}ff.jpg".format(name_idx - 1)))
            save_image(flow2img(results_dict['flowt1_pred_list'][0]),
                       os.path.join(out_path + '_flow', "{:0>7d}bb.jpg".format(name_idx - 1)))
            if "flowfwd_pre" in results_dict.keys():
                save_image(flow2img(results_dict['flowfwd_pre']),
                           os.path.join(out_path + '_flow', "pre_{:0>7d}ff.jpg".format(name_idx - 1)))
            # save_image(results_dict['refine_res'], os.path.join(out_path, "refine_res.jpg"))
            # save_image(results_dict['refine_mask'][-1], os.path.join(out_path, f"refine_mask_{name_idx - 1:0>7d}.jpg"))
            # save_image(results_dict['warped_img0'], os.path.join(out_path, f"warped_img0_{name_idx - 1:0>7d}.jpg"))
            # save_image(results_dict['warped_img1'], os.path.join(out_path, f"warped_img1_{name_idx - 1:0>7d}.jpg"))
            # save_image(results_dict['merged_img'], os.path.join(out_path, "merged_img.jpg"))
            # save_image(results_dict['pre_interp'], os.path.join(out_path, "pre_interp.jpg"))
            # save_image((img0 +img4) / 2, os.path.join(out_path, "overlayed_img.jpg"))

            img_pred = imgt_pred
            # img_pred = F.interpolate(img_pred, scale_factor=1 // pre_down, mode='bilinear')
            cv2.imwrite(out_path + '/{:0>7d}.jpg'.format(name_idx),
                        (img_pred[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1])
            # for j in range(len(results_dict['flowt0_pred_list'])):
            #     # cv2.imwrite(out_path + f'/down_{j}_{name_idx:0>7d}.jpg',
            #     #             (imgt_preds[j][0].clip(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1])
            #     save_image(flow2img(results_dict['flowt0_pred_list'][j]),
            #                os.path.join(out_path + '_flow', f"down_{j}_{name_idx:0>7d}.jpg"))
            name_idx += 1
        # img4 = F.interpolate(img4, scale_factor=1 // pre_down, mode='bilinear')
        cv2.imwrite(out_path + '/{:0>7d}.jpg'.format(name_idx),
                    (img4[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1])
        name_idx += 1
        idx += 1
        img0 = img4