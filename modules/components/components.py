import copy

components = {}


def register(name):
    def decorator(cls):
        components[name] = cls
        return cls
    return decorator


def make_components(model_spec, args=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = components[model_spec['name']](**model_args)
    return model
