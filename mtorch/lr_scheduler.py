import torch

class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        print("optimizer base lr: {}".format(self.base_lrs))
        self.step(last_iter + 1)
        self.last_iter = last_iter

    def state_dict(self):
        return {key: value for key, value in self.__dict__items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, iteration=None):
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class LinearDecreasingLR(_LRScheduler):
    def __init__(self, optimizer, total_iter, last_iter=-1):
        self.total_iter = total_iter
        super(LinearDecreasingLR, self).__init__(optimizer, last_iter)

    def get_lr(self):
        return [base_lr * (1.0 - 1.0 * self.last_iter / self.total_iter) for base_lr in self.base_lrs]
