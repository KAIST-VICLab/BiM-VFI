import torch
import logging
import os
import os.path as osp
import time
import datetime
import random
import wandb
import yaml
import json
import numpy as np

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

import utils
from utils.misc import print_cuda_statistics, is_main_process, get_rank, get_world_size
import datasets
import modules.models as models


class Trainer(object):
    """
    Wrapper for training, more related to engineering than research code
    """

    def __init__(self, cfgs):
        self.rank = get_rank()
        self.cfgs = cfgs
        self.is_master = (self.rank == 0)
        self.is_train = False

        env = cfgs['env']
        self.tot_gpus = get_world_size()
        self.distributed = (get_world_size() > 1)

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger = utils.misc.set_save_dir(cfgs['log_dir'], cfgs["run_description"], replace=False)
            with open(osp.join(cfgs['cfg_dir'], f'cfg_{cfgs["run_description"]}.yaml'), 'w') as f:
                yaml.dump(cfgs, f, sort_keys=False)

            self.log = logger.info

            self.enable_tb = True


        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        self.make_datasets()
        self.model = models.make(cfgs)
        self.start_epoch = 0
        self.end_epoch = self.cfgs['max_epoch']
        if 'resume' in self.cfgs:
            run_id = self.model.load_checkpoint(self.cfgs['resume'])
            self.start_epoch = self.model.current_epoch
        else:
            run_id = wandb.util.generate_id()
        if self.is_master and env['wandb_upload']:
            self.enable_wandb = True
            self.cfgs['enable_wandb'] = True
            with open('wandb.yaml', 'r') as f:
                wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
            os.environ['WANDB_DIR'] = env['save_dir']
            os.environ['WANDB_NAME'] = env['exp_name']
            os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
            wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfgs, id=run_id, name=env['exp_name'],
                       resume='allow')
        else:
            self.enable_wandb = False
            self.cfgs['enable_wandb'] = False

    def make_datasets(self):
            """
                By default, train dataset performs shuffle and drop_last.
                Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

                Cfg example:

                train/test_dataset:
                    name:
                    args:
                    loader: {batch_size: , num_workers: }
            """
            cfgs = self.cfgs
            self.dist_samplers = []

            def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
                sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
                loader = DataLoader(
                    dataset,
                    batch_size // self.tot_gpus,
                    drop_last=drop_last,
                    sampler=sampler,
                    shuffle=(shuffle and (sampler is None)),
                    num_workers=num_workers // self.tot_gpus,
                    pin_memory=True)
                return loader, sampler

            if cfgs.get('train_dataset') is not None:
                train_dataset = datasets.make(cfgs['train_dataset'])
                self.log(f'Train dataset: len={len(train_dataset)}')
                l = cfgs['train_dataset']['loader']
                self.train_loader, train_sampler = make_distributed_loader(
                    train_dataset, l['batch_size'], l['num_workers'], shuffle=True, drop_last=True)
                self.dist_samplers.append(train_sampler)
                self.cfgs['lr_scheduler']['args']['total_steps'] = len(self.train_loader) * self.cfgs['max_epoch']

            if cfgs.get('test_dataset') is not None:
                test_dataset = datasets.make(cfgs['test_dataset'])
                self.log(f'Test dataset: len={len(test_dataset)}')
                l = cfgs['test_dataset']['loader']
                self.test_loader, test_sampler = make_distributed_loader(
                    test_dataset, l['batch_size'], l['num_workers'], shuffle=False, drop_last=False)
                self.dist_samplers.append(test_sampler)
            if cfgs.get('demo_dataset') is not None:
                self.demo_root = self.cfgs['demo_dataset']['args']['root_path']

    def train(self):
        print("Start training")
        start_time = time.time()
        self.is_train = True
        self.model.init_training_logger()
        self.best_performance = 0
        # torch.backends.cudnn.benchmark = True
        for epoch in range(self.start_epoch, self.end_epoch):
            if self.cfgs['distributed']:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            random.seed(self.cfgs['seed'] + epoch)
            np.random.seed(self.cfgs['seed'] + epoch)
            torch.random.manual_seed(self.cfgs['seed'] + epoch)
            torch.manual_seed(self.cfgs['seed'] + epoch)
            torch.cuda.manual_seed_all(self.cfgs['seed'] + epoch)

            self.model.train_one_epoch(self.train_loader, epoch)

            if ((epoch + 1) % self.cfgs['validate_every']) == 0:
                if is_main_process():
                    performance = self.validate()
                    if performance > self.best_performance:
                        self.best_performance = performance
                        self.model.save_checkpoint('model_{}.pth'.format(epoch + 1), is_best=1)
                        self.log(
                            "best performance achieved at epoch {} with performance of {}".format(epoch,
                                                                                                  self.best_performance))

            if ((epoch + 1) % self.cfgs['save_every']) == 0 and is_main_process():
                self.model.save_checkpoint('model_{}.pth'.format(epoch + 1))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        if is_main_process():
            self.finalize_training()

    def validate(self):
        # return performance to save the best model, if there is no performance measure e.g. GAN then just return 0
        if not self.is_train:  # if mode == validation only
            self.model.init_validation_logger()
        return self.model.validate(self.test_loader)

    def benchmark(self):
        self.model.init_testing_logger()
        self.model.benchmark()


    def demo(self):
        self.model.init_demo_logger()
        self.model.demo(self.demo_root)

    def finalize_training(self):
        self.model.finalize_training()
