from torch.optim.lr_scheduler import *


def make_lr_scheduler(optimizer, lr_scheduler_spec):
    lr_scheduler = {
        'step_lr': StepLR,
        'one_cycle_lr': OneCycleLR,
        'cosine_lr': CosineAnnealingLR,
        'constant_lr': ConstantLR,

    }[lr_scheduler_spec['name']](optimizer, **lr_scheduler_spec['args'])
    return lr_scheduler
