import logging
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR


class MultiFixedScheduler(MultiStepLR):
    def __init__(self, optimizer, stageiters, stagelrs, last_iter=-1):
        assert isinstance(stageiters, (list, tuple, np.ndarray))
        assert isinstance(stagelrs, (list, tuple, np.ndarray))
        assert len(stagelrs) == len(stageiters)
        assert len(stageiters) > 0

        self.steps = np.array(stageiters)
        assert (self.steps[0] >= 0) and np.all(np.diff(self.steps) > 0)
        self.lrs = np.array(stagelrs)
        assert np.all(self.lrs > 0)
        self.last_lr = None
        super(MultiFixedScheduler, self).__init__(optimizer, stageiters, last_epoch=last_iter)

    def get_lr(self):
        idx = np.searchsorted(self.steps, self.last_epoch)

        if idx >= len(self.lrs):
            idx = len(self.lrs) - 1
        lr = self.lrs[idx]
        return lr

    def step(self, iterations=None):
        if iterations is None:
            iterations = self.last_epoch + 1
        self.last_epoch = iterations
        new_lr = self.get_lr()
        if self.last_lr == new_lr:
            return
        logging.info("Iteration {}, lr = {}".format(iterations, new_lr))
        self.last_lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
