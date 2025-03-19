from torch.optim import *


def make_optimizer(params, optimizer_spec):
    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamW': AdamW
    }[optimizer_spec['name']](params, **optimizer_spec['args'])
    return optimizer
