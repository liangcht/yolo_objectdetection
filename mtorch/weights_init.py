import torch.nn as nn
from mtorch.caffetorch import Scale


def msra_init(net):
    """Init layer parameters.
    :param net: nn.Module or nn.Sequential
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in') 
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, Scale):
            m.weight.data.fill_(1)  # as opposed to uniform initialization by default
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):  # MSRA initialization is different for:
            m.running_var.data.zero_()  # as opposed to unit initialization by default
            if m.weight is not None:
                m.weight.data.fill_(1)  # as opposed to uniform initialization by default
            if m.bias is not None:
                m.bias.data.zero_()  # as opposed to uniform initialization by default
