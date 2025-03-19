import argparse
import os

import yaml
import torch
import torch.multiprocessing as mp

import utils
from utils.experiment import *
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    parser.add_argument('--load-root', default='data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--port-offset', '-p', type=int, default=0)
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    args = parser.parse_args()

    return args


def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                d[k] = v.replace('$load_root$', args.load_root)
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0].replace('_benchmark', '').replace('_demo', '')
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag

    env = dict()
    env['exp_name'] = exp_name + '_' + cfg['exp_name']
    env['save_dir'] = os.path.join(args.save_root, env['exp_name'])
    env['tot_gpus'] = torch.cuda.device_count()
    env['cudnn'] = args.cudnn
    env['port'] = str(29600 + args.port_offset)
    env['wandb_upload'] = args.wandb_upload
    cfg['env'] = env

    return cfg


def main():
    args = parse_args()

    cfgs = make_cfg(args)

    init_experiment(cfgs)
    init_distributed_mode(cfgs)
    init_deterministic(cfgs['seed'])

    trainer = Trainer(cfgs)

    if cfgs['mode'] == 'train':
        trainer.train()
    elif cfgs['mode'] == 'validate':
        trainer.validate()
    elif cfgs['mode'] == 'benchmark':
        trainer.benchmark()
    elif cfgs['mode'] == 'demo':
        trainer.demo()



if __name__ == '__main__':
    main()
