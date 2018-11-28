import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

import region_target_cuda
import region_target_cpu


class RegionTargetFunction(Function):
    @staticmethod
    def forward(ctx, xy, wh, obj, truth,
                biases,
                coord_scale, positive_thresh,
                warmup, rescore):
        if xy.is_cuda:
            rt_ = region_target_cuda
        else:
            rt_ = region_target_cpu
        return tuple(rt_.forward(
            xy, wh, obj, truth,
            biases,
            coord_scale, positive_thresh,
            warmup, rescore
        ))

    @staticmethod
    def backward(ctx, *args):
        return tuple([None] * 9)


class RegionTarget(nn.Module):
    def __init__(self, biases=None, rescore=True, anchor_aligned_images=12800, coord_scale=1.0, positive_thresh=0.6,
                 gpus_size=1, seen_images=0):
        super(RegionTarget, self).__init__()
        if biases is None:
            biases = []
        self.rescore = rescore  # type: bool
        self.coord_scale = coord_scale
        self.positive_thresh = positive_thresh
        self.anchor_aligned_images = anchor_aligned_images
        self.gpus_size = gpus_size
        self.register_buffer('biases', torch.from_numpy(np.array(biases, dtype=np.float32)))
        # noinspection PyCallingNonCallable,PyUnresolvedReferences
        self.register_buffer('seen_images', torch.tensor(seen_images, dtype=torch.long))

    def forward(self, xy, wh, obj, truth):
        self.seen_images += xy.size(0) * self.gpus_size
        warmup = self.seen_images.item() < self.anchor_aligned_images
        return RegionTargetFunction.apply(
            xy.detach(), wh.detach(), obj.detach(), truth.detach(),
            self.biases,
            self.coord_scale, self.positive_thresh,
            warmup, self.rescore
        )

    def extra_repr(self):
        """Extra information
        """
        return '{}biases={}{}'.format(
            "rescore, " if self.rescore else "", [round(b.item(), 3) for b in self.biases.view(-1)],
            ", seen_images={}".format(self.seen_images.item()) if self.seen_images.item() else ""
        )
