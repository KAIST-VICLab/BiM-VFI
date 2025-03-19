import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(cfgs):
    model = models[cfgs['model']['name']](cfgs)
    return model
