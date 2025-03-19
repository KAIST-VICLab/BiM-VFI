import logging
import os
import shutil
from tensorboardX import SummaryWriter
from typing import Iterable
from pathlib import Path
from time import time
import datetime
import wandb
import ptflops
import pyiqa

import lpips
import stlpips_pytorch.stlpips as stlpips

import torch
from torchvision.utils import save_image

import utils.misc
import utils.misc as misc
from utils.plot import plot_samples_per_epoch
from utils.metrics import calculate_batch_psnr, calculate_batch_ssim, calculate_batch_lpips, calculate_batch_stlpips, calculate_batch_niqe
from utils.flowvis import flow2img
from modules.components import make_components
from modules.loss import make_loss_dict
from modules.lr_scheduler import make_lr_scheduler
from modules.optimizer import make_optimizer
from modules.models import register
from modules.models.inference_video import inference_demo
import datasets


@register('base_model')
class BaseModel:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = torch.cuda.current_device()

        self.current_iteration = 0
        self.current_epoch = 0
        self.model = make_components(self.cfgs['model'])
        self.lpips_metric = lpips.LPIPS(net="alex")
        self.stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")
        self.niqe_metric = pyiqa.create_metric('niqe', device=self.device)

        self.logger = logging.getLogger(self.cfgs['model']['name'])
        if self.cfgs['mode'] == 'train':
            self.loss_dict = make_loss_dict(cfgs['loss'])
        self.move_components_to_device(cfgs['mode'])

        self.model_without_ddp = self.model
        if cfgs['distributed']:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[cfgs['gpu']])
            self.model_without_ddp = self.model.module
        if self.cfgs['mode'] == 'train':
            self.optimizer = make_optimizer(self.model_without_ddp.parameters(), self.cfgs['optimizer'])
            self.lr_scheduler = make_lr_scheduler(self.optimizer, cfgs['lr_scheduler'])
            self.train_one_step_opt = self.train_one_step
        print(f'Total params: {self.count_parameters()}')

    def load_checkpoint(self, file_path):
        """
        Load checkpoint
        """
        checkpoint = torch.load(file_path, map_location="cpu")

        self.current_epoch = checkpoint['epoch']
        self.current_iteration = checkpoint['iteration']
        self.model_without_ddp.load_state_dict(checkpoint['model'])
        if self.cfgs['mode'] == 'train':
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.logger.info('Chekpoint loaded successfully from {} at epoch: {} and iteration: {}'.format(
            file_path, checkpoint['epoch'], checkpoint['iteration']))
        self.move_components_to_device(self.cfgs['mode'])
        return checkpoint['run_id']


    def save_checkpoint(self, file_name, is_best=0):
        """
        Save checkpoint
        """
        state = {
            'epoch': self.current_epoch,  # because epoch is used for loading then this must be added + 1
            'iteration': self.current_iteration,
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'run_id': wandb.run.id if wandb.run is not None else -1
        }

        misc.save_on_master(state, os.path.join(self.cfgs['checkpoint_dir'], file_name))

        if is_best and misc.is_main_process():
            shutil.copyfile(os.path.join(self.cfgs['checkpoint_dir'], file_name),
                            os.path.join(self.cfgs['checkpoint_dir'], 'model_best.pth'))

    def adjust_learning_rate(self, epoch):
        """
        Adjust learning rate every epoch
        """
        self.lr_scheduler.step()

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
                flowfwd = flow2img(result_dict['flowfwd'][:nsample].detach())
                flowfwd_gt = flow2img(input_dict['flowt0'][:nsample])

                figure = torch.stack(
                    [overlapped_img, imgt_pred_p, flowfwd, flowfwd_gt, gt_p])
                figure = torch.transpose(figure, 0, 1).reshape(-1, 3, patch_size, patch_size)
                image = plot_samples_per_epoch(
                    figure, os.path.join(self.cfgs['output_dir'], "imgs_train"),
                    self.current_epoch, self.current_iteration, nsample
                )

                if self.cfgs['enable_wandb']:
                    wandb.log({"loss": loss}, step=self.current_iteration)
                    for k, v in losses.items():
                        wandb.log({f'loss_{k}': v}, step=self.current_iteration)
                    wandb.log({"lr": torch.Tensor(self.lr_scheduler.get_last_lr())},
                              step=self.current_iteration)
                    if self.current_iteration % (print_freq * 10) == 0:
                        wandb.log({"Image": wandb.Image(image)}, step=self.current_iteration)
                else:
                    self.summary_writer.add_image("Train/image", image, self.current_iteration)
                    self.summary_writer.add_scalar("Train/loss", loss, self.current_iteration)
                    for k, v in losses.items():
                        self.summary_writer.add_scalar(f'Train/loss_{k}', v, self.current_iteration)
                    self.summary_writer.add_scalar("Train/LR", self.lr_scheduler.get_last_lr(), self.current_iteration)

            self.current_iteration += 1

            # gather the stats from all processes
        self.metric_logger.synchronize_between_processes()
        self.current_epoch += 1
        if utils.misc.is_main_process():
            self.logger.info(f"Averaged training stats: {self.metric_logger}")

    def train_one_step(self, input_dict):
        """
        Training step for each mini-batch
        """
        result_dict = self.model(**input_dict)
        loss = torch.Tensor([0]).to(self.device)
        losses = dict()
        for k, v in self.loss_dict.items():
            losses[k] = v(**result_dict, **input_dict, epoch=self.current_epoch).mean()
            loss += losses[k]
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if 'gradient_clip' in self.cfgs:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfgs['gradient_clip'])
        self.optimizer.step()
        return loss, losses, result_dict

    @torch.no_grad()
    def validate(self, val_loader, test_indicator=None):
        """
        Validation step for each mini-batch
        """
        self.model.eval()

        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        self.metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Test:'
        psnr_dict = {}

        print_freq = 10
        if test_indicator is None:
            test_indicator = f"{self.cfgs['test_dataset']['name']}_{self.cfgs['test_dataset']['args']['split']}"

        for input_dict in self.metric_logger.log_every(val_loader, print_freq, header):
            input_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
            img0 = input_dict['img0']
            imgt = input_dict['imgt']
            img1 = input_dict['img1']
            result_dict = self.model(**input_dict, run_with_gt=True)

            scene_names = input_dict['scene_names']

            imgt_pred = result_dict['imgt_pred']
            warped_img0 = result_dict['warped_img0']
            warped_img1 = result_dict['warped_img1']
            refine_mask = result_dict['refine_mask'][-1]
            psnr, psnr_list = calculate_batch_psnr(imgt, imgt_pred)
            ssim, bs = calculate_batch_ssim(imgt, imgt_pred)
            self.metric_logger.update(psnr={'value': psnr, 'n': len(psnr_list)},
                                      ssim={'value': ssim, 'n': len(psnr_list)})
            if (self.cfgs['mode'] == 'train' and self.current_epoch % self.cfgs['vis_every'] == 0) or (
                    self.cfgs['mode'] != 'train' and self.cfgs['test_dataset']['save_imgs']):
                for i in range(len(scene_names)):
                    psnr_dict[scene_names[i]] = float(psnr_list[i])
                    if self.cfgs['mode'] == "test":
                        scene0_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[0])
                        scenet_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[1])
                        scene1_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[2])
                    else:
                        scene0_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[0])
                        scenet_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[1])
                        scene1_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[2])
                    Path(scene0_path).parent.mkdir(exist_ok=True, parents=True)
                    save_image(img0[0], scene0_path)
                    save_image(imgt_pred[0], scenet_path.replace('.', '_pred.'))
                    save_image(imgt[0], scenet_path)
                    save_image(img1[0], scene1_path)
                    save_image(warped_img0[0], scenet_path.replace('.', '_warped_0.'))
                    save_image(warped_img1[0], scenet_path.replace('.', '_warped_1.'))
                    save_image(refine_mask[0], scenet_path.replace('.', '_refine.'))
                    save_image((img1[0] + img0[0]) / 2, scenet_path.replace('.', '_overlayed.'))
                    save_image(flow2img(result_dict['flowfwd'])[0], scenet_path.replace('.', '_flow_t1.'))
                    save_image(flow2img(result_dict['flowbwd'])[0], scenet_path.replace('.', '_flow_t0.'))
        # gather the stats from all processes
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
            wandb.log({'val_psnr': self.metric_logger.psnr.global_avg, 'val_ssim': self.metric_logger.ssim.global_avg},
                      step=self.current_iteration)
        if self.cfgs['mode'] == "test" and self.cfgs['enable_wandb']:
            wandb.run.summary[
                f"{test_indicator}_psnr"] = self.metric_logger.psnr.global_avg
            wandb.run.summary[
                f"{test_indicator}_ssim"] = self.metric_logger.ssim.global_avg
        return self.metric_logger.psnr.global_avg

    @torch.no_grad()
    def demo(self, video_dir):
        start_time = time()
        for video_name in os.listdir(video_dir):
            # video_name = "Awesome_Again_Stakes_2019.mkv"
            video_path = os.path.join(video_dir, video_name)
            out_path = os.path.join(self.cfgs['output_dir'], 'demo', video_name.split(".")[0])
            inference_demo(self.model, 8, video_path, out_path)
        total_time_str = str(datetime.timedelta(seconds=int(time() - start_time)))
        print("Total time: {}".format(total_time_str))

    def init_training_logger(self):
        """
        Initialize training logger specific for each model
        """
        if misc.is_main_process():
            self.summary_writer = SummaryWriter(log_dir=self.cfgs['summary_dir'], comment='m2mpwc')
            Path(os.path.join(self.cfgs['output_dir'], 'imgs_train')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(self.cfgs['output_dir'], 'imgs_val')).mkdir(parents=True, exist_ok=True)
        self._reset_metric()

    def init_validation_logger(self):
        """
        Initialize validation logger specific for each model
        """
        if misc.is_main_process():
            self.summary_writer = SummaryWriter(log_dir=self.cfgs['summary_dir'], comment='m2mpwc')
            Path(os.path.join(self.cfgs['output_dir'], 'imgs_val')).mkdir(parents=True, exist_ok=True)
        self._reset_metric()

    def init_testing_logger(self):
        """
        Initialize testing logger specific for each model
        """
        if misc.is_main_process():
            self.summary_writer = SummaryWriter(log_dir=self.cfgs['summary_dir'], comment='m2mpwc')
            Path(os.path.join(self.cfgs['output_dir'], 'imgs_test')).mkdir(parents=True, exist_ok=True)
        self._reset_metric()

    def init_demo_logger(self):
        """
        Initialize testing logger specific for each model
        """
        if misc.is_main_process():
            self.summary_writer = SummaryWriter(log_dir=self.cfgs['summary_dir'], comment='m2mpwc')
            Path(os.path.join(self.cfgs['output_dir'], 'demo')).mkdir(parents=True, exist_ok=True)
        self._reset_metric()

    @torch.no_grad()
    def finalize_training(self):
        self.benchmark()

    @torch.no_grad()
    def benchmark(self):
        self.model.eval()
        dataset_list = self.cfgs['benchmark_dataset']['name']
        root_list = self.cfgs['benchmark_dataset']['args']['root_path']
        split_list = self.cfgs['benchmark_dataset']['args']['split']
        pyr_lvl_list = self.cfgs['benchmark_dataset']['args']['pyr_level']
        for i, dataset in enumerate(dataset_list):
            for split in split_list[i]:
                arg_dict = {'name': dataset, 'args': {'root_path': root_list[i], 'split': split}}
                val_dataset = datasets.make(arg_dict)
                self.metric_logger = misc.MetricLogger(delimiter="  ")
                self.metric_logger.add_meter('psnr', misc.SmoothedValue(window_size=1))
                self.metric_logger.add_meter('ssim', misc.SmoothedValue(window_size=1))
                self.metric_logger.add_meter('lpips', misc.SmoothedValue(window_size=1))
                self.metric_logger.add_meter('stlpips', misc.SmoothedValue(window_size=1))
                self.metric_logger.add_meter('niqe', misc.SmoothedValue(window_size=1))
                header = 'Test:'

                print_freq = 10
                test_indicator = f"{dataset}_{split}"
                psnr_dict = {}
                lpips_dict = {}
                stlpips_dict = {}

                for input_dict in self.metric_logger.log_every(val_dataset, print_freq, header):
                    input_dict = {k: v.to(self.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
                    img0 = input_dict['img0']
                    img1 = input_dict['img1']
                    imgt = input_dict['imgt']
                    scene_names = input_dict['scene_names']
                    result_dict = self.model(pyr_level=pyr_lvl_list[i], **input_dict)
                    imgt_pred = result_dict['imgt_pred'].clip(0, 1)
                    imgt_pred = (imgt_pred * 255.).round() / 255.
                    torch.cuda.empty_cache()
                    psnr, psnr_list = calculate_batch_psnr(imgt, imgt_pred)
                    ssim, bs = calculate_batch_ssim(imgt, imgt_pred)
                    lpips, bs = calculate_batch_lpips(imgt, imgt_pred, self.lpips_metric)
                    stlpips, bs = calculate_batch_stlpips(imgt, imgt_pred, self.stlpips_metric)
                    niqe, bs = calculate_batch_niqe(imgt, imgt_pred, self.niqe_metric)
                    self.metric_logger.update(psnr={'value': psnr, 'n': len(psnr_list)},
                                              ssim={'value': ssim, 'n': len(psnr_list)},
                                              lpips={'value': lpips, 'n': len(psnr_list)},
                                              stlpips={'value': stlpips, 'n': len(psnr_list)},
                                              niqe={'value': niqe, 'n': len(psnr_list)},
                                              )
                    # gather the stats from all processes
                    if (self.cfgs['mode'] == 'train' and self.current_epoch % self.cfgs['vis_every'] == 0) or (
                    self.cfgs['mode'] != 'train' and self.cfgs['benchmark_dataset']['save_imgs']):
                        psnr_dict[scene_names[1]] = float(psnr_list[0])
                        lpips_dict[scene_names[1]] = float(lpips)
                        stlpips_dict[scene_names[1]] = float(stlpips)
                        if self.cfgs['mode'] == "test":
                            scene0_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[0])
                            scenet_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[1])
                            scene1_path = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator, scene_names[2])
                        else:
                            scene0_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[0])
                            scenet_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[1])
                            scene1_path = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator, scene_names[2])
                        Path(scene0_path).parent.mkdir(exist_ok=True, parents=True)
                        save_image(img0[0], scene0_path)
                        save_image(imgt_pred[0], scenet_path.replace('.', '_pred.'))
                        save_image(imgt[0], scenet_path)
                        save_image(img1[0], scene1_path)
                        save_image((img1[0] + img0[0]) / 2, scenet_path.replace('.', '_overlayed.'))
                self.logger.info(f"Averaged validate stats:{self.metric_logger.print_avg()}")
                if (self.cfgs['mode'] == 'train' and self.current_epoch % self.cfgs['vis_every'] == 0) or (
                        self.cfgs['mode'] != 'train' and self.cfgs['benchmark_dataset']['save_imgs']):
                    psnr_str = []
                    psnr_dict = sorted(psnr_dict.items(), key=lambda item: item[1])
                    for key, val in psnr_dict:
                        psnr_str.append("{}: {} {} {}".format(key, val, lpips_dict[key], stlpips_dict[key]))
                    psnr_str = "\n".join(psnr_str)
                    if self.cfgs['mode'] == "test":
                        outdir = os.path.join(self.cfgs['output_dir'], "imgs_test", test_indicator)
                    else:
                        outdir = os.path.join(self.cfgs['output_dir'], "imgs_val", test_indicator)
                    with open(os.path.join(outdir, "results.txt"), "w") as f:
                        f.write(psnr_str)
                if self.cfgs['enable_wandb']:
                    wandb.run.summary[
                        f"{test_indicator}_psnr"] = self.metric_logger.psnr.global_avg
                    wandb.run.summary[
                        f"{test_indicator}_ssim"] = self.metric_logger.ssim.global_avg
                    wandb.run.summary[
                        f"{test_indicator}_lpips"] = self.metric_logger.lpips.global_avg
                    wandb.run.summary[
                        f"{test_indicator}_stlpips"] = self.metric_logger.stlpips.global_avg
                    wandb.run.summary[
                        f"{test_indicator}_niqe"] = self.metric_logger.niqe.global_avg

    def move_components_to_device(self, mode):
        """
        Move components to device
        """
        self.model.to(self.device)
        self.lpips_metric.to(self.device)
        self.stlpips_metric.to(self.device)
        if self.cfgs['mode'] == 'train':
            for _, v in self.loss_dict.items():
                v.to(self.device)
        self.logger.info('Model: {}'.format(self.model))

    def _reset_metric(self):
        """
        Metric related to average meter
        """
        self.metric_logger = misc.MetricLogger(delimiter="  ")
        self.metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        self.metric_logger.add_meter('loss', misc.SmoothedValue(window_size=20))

    def count_parameters(self):
        """
        Return the number of parameters for the model
        """
        model_number = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)

        def input_constructor(input_shape):
            inputs = {'img0': torch.rand(*input_shape).cuda(), 'img1': torch.rand(*input_shape).cuda(), 'time_step': torch.tensor([0.5]).cuda(), 'pyr_level': 5}
            return inputs
        self.model.eval()
        flops = ptflops.get_model_complexity_info(self.model, (1, 3, 256, 256), print_per_layer_stat=True, input_constructor=input_constructor)
        return model_number
