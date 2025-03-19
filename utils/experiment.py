"""
Experiment related stuffs
Act as a bridge between main and utils (logging, init directory, etc)
"""
from pathlib import Path
import os
import random
import numpy as np
# import cupyx.distributed

import torch.distributed as dist
import torch


def init_experiment(cfgs):
    """
    in:
        cfgs: arguments such as hyperparameters and other
    out:
        --cfgs
    procedure to initialize experiment consisting of:
        - parse config file as a json dictionary
        - initialize logging
        - create dictionary to save everything
    """

    assert 'exp_name' in cfgs

    cfgs['summary_dir'] = os.path.join(cfgs['env']['save_dir'], "summaries")
    cfgs['checkpoint_dir'] = os.path.join(cfgs['env']['save_dir'], "checkpoints")
    cfgs['output_dir'] = os.path.join(cfgs['env']['save_dir'], "output")
    cfgs['log_dir'] = os.path.join(cfgs['env']['save_dir'], "logs")
    cfgs['cfg_dir'] = os.path.join(cfgs['env']['save_dir'], "cfgs")
    mode = cfgs["mode"]
    if mode == "demo":
        dataset = cfgs[f"{mode}_dataset"]['name']
        cfgs['run_description'] = f'{mode}_{dataset}'
    else:
        dataset = cfgs[f"{mode}_dataset"]['name']
        split = cfgs[f"{mode}_dataset"]['args']['split']
        cfgs['run_description'] = f'{mode}_{dataset}_{split}'

    Path(cfgs['summary_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfgs['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfgs['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfgs['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(cfgs['cfg_dir']).mkdir(parents=True, exist_ok=True)


def init_deterministic(random_seed=7):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = True


def init_distributed_mode(cfgs):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfgs['rank'] = int(os.environ["RANK"])
        cfgs['world_size'] = int(os.environ['WORLD_SIZE'])
        cfgs['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        cfgs['rank'] = int(os.environ['SLURM_PROCID'])
        cfgs['gpu'] = cfgs['rank'] % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        cfgs['distributed'] = False
        return

    cfgs['distributed'] = True

    torch.cuda.set_device(cfgs['gpu'])
    cfgs['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        cfgs['rank'], cfgs['dist_url']), flush=True)
    dist.init_process_group(backend=cfgs['dist_backend'], init_method=cfgs['dist_url'],
                            world_size=cfgs['world_size'], rank=cfgs['rank'])
    # cupyx.distributed.NCCLBackend(n_devices=cfgs['world_size'], rank=cfgs['rank'])
    dist.barrier()
