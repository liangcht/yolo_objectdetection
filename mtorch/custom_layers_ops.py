import torch.nn as nn


def _set_bn_eval(m):
    """sets BatchNorm modules to evaluation state"""
    if isinstance(m, nn.BatchNorm2d):
        m.eval()


def _freeze_all_modules(model):
    # gradient will not be tracked
    for param in model.parameters():
        param.requires_grad = False
    # running BN statistics will not be updated
    for module in model.children():
        module.apply(_set_bn_eval)


def _freeze_till(model, layer_name):
    """ freezes till specified layer"""
    # gradient will not be tracked
    freeze_stopped = False
    for name, param in model.named_parameters():
        if layer_name in name:
            freeze_stopped = True
            break
        param.requires_grad = False
    # running BN statistics will not be updated
    for name, module in model.named_modules():
        if layer_name in name:
            freeze_stopped = True
            break
        _set_bn_eval(module)
    return freeze_stopped


def freeze_modules_for_training(model, layer_to_stop_freezing=None):
    """frees modules for training
    :param model: nn.Module or nn.Sequential
    :param layer_to_stop_freezing: name of the layer at which to stop freezing
    :return true if the freezing was stopped at some point
    """
    if not layer_to_stop_freezing:
        _freeze_all_modules(model)
        return True
    return _freeze_till(model, layer_to_stop_freezing)
