from modules.models.base_model import BaseModel
import os
from typing import Iterable
from pathlib import Path
import wandb

import torch
from torchvision.utils import save_image

import utils.misc
import utils.misc as misc
from utils.plot import plot_samples_per_epoch
from utils.metrics import calculate_batch_psnr, calculate_batch_ssim, calculate_batch_lpips, calculate_batch_stlpips, calculate_batch_niqe
from utils.flowvis import flow2img
from modules.models import make, register


@register('bim_vfi')
class BiMVFI(BaseModel):
    def __init__(self, cfg):
        super(BiMVFI, self).__init__(cfg)

    def train_one_epoch(self, train_loader: Iterable, epoch: int, max_norm: float = 0):
        """
        Training of one epoch
        """
        self.current_epoch = epoch
        self._reset_metric()

        self.model.train()

        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 100
        for input_dict in self.metric_logger.log_every(train_loader, print_freq, header):
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
            loss, losses, result_dict = self.train_one_step_opt(input_dict)
            self.lr_scheduler.step()
            imgt_pred = result_dict['imgt_pred']
            imgt_pred = torch.clamp(imgt_pred, 0, 1)

            self.metric_logger.update(loss=loss, **losses)
            self.metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            if misc.is_main_process() and self.current_iteration % print_freq == 0:
                nsample = 4
                patch_size = self.cfgs['train_dataset']['args']['patch_size']

                img0_p, img1_p = input_dict['img0'][:nsample].detach(), input_dict['img1'][:nsample].detach()
                gt_p, imgt_pred_p = input_dict['imgt'][:nsample].detach(), imgt_pred[:nsample].detach()
                overlapped_img = img0_p * 0.5 + img1_p * 0.5

                if self.cfgs['enable_wandb']:
                    wandb.log({"loss": loss}, step=self.current_iteration)
                    for k, v in losses.items():
                        wandb.log({f'loss_{k}': v}, step=self.current_iteration)
                    wandb.log({"lr": torch.Tensor(self.lr_scheduler.get_last_lr())},
                              step=self.current_iteration)
                    if self.current_iteration % (print_freq * 10) == 0:
                        flowfwd = flow2img(result_dict['flowt0_pred_list'][0][:nsample].detach())

                        figure = torch.stack(
                            [overlapped_img, imgt_pred_p, flowfwd, gt_p])
                        figure = torch.transpose(figure, 0, 1).reshape(-1, 3, patch_size, patch_size)
                        image = plot_samples_per_epoch(
                            figure, os.path.join(self.cfgs['output_dir'], "imgs_train"),
                            self.current_epoch, self.current_iteration, nsample, 'train'
                        )
                        wandb.log({"Image": wandb.Image(image, file_type="jpg")}, step=self.current_iteration)
                else:
                    self.summary_writer.add_scalar("Train/loss", loss, self.current_iteration)
                    for k, v in losses.items():
                        self.summary_writer.add_scalar(f'Train/loss_{k}', v, self.current_iteration)
                    self.summary_writer.add_scalar("Train/LR", self.lr_scheduler.get_last_lr(), self.current_iteration)
                    if self.current_iteration % (print_freq * 10) == 0:
                        flowfwd = flow2img(result_dict['flowt0_pred_list'][0][:nsample].detach())

                        figure = torch.stack(
                            [overlapped_img, imgt_pred_p, flowfwd, gt_p])
                        figure = torch.transpose(figure, 0, 1).reshape(-1, 3, patch_size, patch_size)
                        image = plot_samples_per_epoch(
                            figure, os.path.join(self.cfgs['output_dir'], "imgs_train"),
                            self.current_epoch, self.current_iteration, nsample, 'train'
                        )
                        self.summary_writer.add_image("Train/image", image, self.current_iteration)

            self.current_iteration += 1
            break

            # gather the stats from all processes
        self.metric_logger.synchronize_between_processes()
        self.current_epoch += 1
        if utils.misc.is_main_process():
            self.logger.info(f"Averaged training stats: {self.metric_logger}")

    @torch.no_grad()
    def validate(self, val_loader, test_indicator=None):
        """
        Validation step for each mini-batch
        """
        self.model.eval()

        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        self.metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        self.metric_logger.add_meter('lpips', misc.SmoothedValue(window_size=1))
        self.metric_logger.add_meter('stlpips', misc.SmoothedValue(window_size=1))
        self.metric_logger.add_meter('niqe', misc.SmoothedValue(window_size=1))
        header = 'Test:'
        psnr_dict = {}
        lpips_dict = {}
        stlpips_dict = {}

        print_freq = 10
        if test_indicator is None:
            test_indicator = f"{self.cfgs['test_dataset']['name']}_{self.cfgs['test_dataset']['args']['split']}"

        for input_dict in self.metric_logger.log_every(val_loader, print_freq, header):
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
            img0 = input_dict['img0']
            imgt = input_dict['imgt']
            img1 = input_dict['img1']
            result_dict = self.model(**input_dict, run_with_gt=False)

            scene_names = input_dict['scene_names']

            imgt_pred = result_dict['imgt_pred']
            psnr, psnr_list = calculate_batch_psnr(imgt, result_dict['imgt_pred'])
            ssim, bs = calculate_batch_ssim(imgt, result_dict['imgt_pred'])
            lpips, bs = calculate_batch_lpips(imgt, imgt_pred, self.lpips_metric)
            stlpips, bs = calculate_batch_stlpips(imgt, imgt_pred, self.stlpips_metric)
            niqe, bs = calculate_batch_niqe(imgt, imgt_pred, self.niqe_metric)
            self.metric_logger.update(psnr={'value': psnr, 'n': len(psnr_list)},
                                      ssim={'value': ssim, 'n': len(psnr_list)},
                                      lpips={'value': lpips, 'n': len(psnr_list)},
                                      stlpips={'value': stlpips, 'n': len(psnr_list)},
                                      niqe={'value': niqe, 'n': len(psnr_list)},
                                      )
            if (self.cfgs['mode'] == 'train' and self.current_epoch % self.cfgs['vis_every'] == 0) or (
                    self.cfgs['mode'] != 'train' and self.cfgs['test_dataset']['save_imgs']):
                for i in range(len(scene_names[1])):
                    psnr_dict[scene_names[1][i]] = float(psnr_list[i])
                    lpips_dict[scene_names[1][i]] = float(lpips)
                    stlpips_dict[scene_names[1][i]] = float(stlpips)
                if self.cfgs['mode'] == "test":
                    scene0_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[0][i])
                    scenet_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[1][i])
                    scene1_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[2][i])
                else:
                    scene0_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[0][i])
                    scenet_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[1][i])
                    scene1_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[2][i])
                    Path(scene0_path).parent.mkdir(exist_ok=True, parents=True)
                save_image(img0[0], scene0_path)
                save_image(imgt_pred[0], scenet_path.replace('.', '_pred.'))
                save_image(imgt[0], scenet_path)
                save_image(img1[0], scene1_path)
                save_image((img1[0] + img0[0]) / 2, scenet_path.replace('.', '_overlayed.'))

        self.logger.info(f"Averaged validate stats:{self.metric_logger.print_avg()}")
        if (self.cfgs['mode'] == 'train' and self.current_epoch % self.cfgs['vis_every'] == 0) or (
                self.cfgs['mode'] != 'train' and self.cfgs['test_dataset']['save_imgs']):
            psnr_str = []
            psnr_dict = sorted(psnr_dict.items(), key=lambda item: item[1])
            for key, val in psnr_dict:
                psnr_str.append("{}: {}".format(key, val))
            psnr_str = "\n".join(psnr_str)
            if self.cfgs['mode'] == "test":
                outdir = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator)
            else:
                outdir = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator)
            with open(os.path.join(outdir, "results.txt"), "w") as f:
                f.write(psnr_str)
        if misc.is_main_process() and self.cfgs['mode'] == 'train':
            self.summary_writer.add_scalar("Val/psnr", self.metric_logger.psnr.global_avg, self.current_epoch)
            self.summary_writer.add_scalar("Val/ssim", self.metric_logger.ssim.global_avg, self.current_epoch)
        if self.cfgs['enable_wandb']:
            wandb.log({'val_psnr': self.metric_logger.psnr.global_avg, 'val_ssim': self.metric_logger.ssim.global_avg,
                       'val_lpips': self.metric_logger.lpips.global_avg, 'val_stlpips': self.metric_logger.stlpips.global_avg,
                       'val_niqe': self.metric_logger.niqe.global_avg},
                      step=self.current_iteration)
        return self.metric_logger.psnr.global_avg
