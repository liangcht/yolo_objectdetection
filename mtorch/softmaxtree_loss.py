import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from mmod.simple_parser import read_softmax_tree

import smtl_cuda
import smtl_cpu
import smt_cuda
import smt_cpu


class SoftmaxTreeWithLossFunction(Function):
    @staticmethod
    def forward(
            ctx,
            x, label,
            group_offsets, group_sizes, cid_groups, parents,
            ignore_label, axis, valid_normalization
    ):
        assert 0 <= axis < x.dim(), "invalid axis for x of size: {}".format(x.size())
        node_count = cid_groups.numel()
        assert x.size(axis) == node_count, "Channel count: {} must match tree node count: {}".format(
            x.size(axis), node_count
        )

        has_ignore_label = ignore_label is not None
        if x.is_cuda:
            smt_ = smt_cuda
            smtl_ = smtl_cuda
        else:
            smt_ = smt_cpu
            smtl_ = smtl_cpu
        prob = smt_.forward(x, group_offsets, group_sizes, axis)[0]
        loss, norm = smtl_.forward(prob, label, parents, has_ignore_label, ignore_label, axis, valid_normalization)
        ctx.ignore_label = ignore_label
        ctx.softmax_axis = axis
        ctx.save_for_backward(prob, label, norm, group_offsets, group_sizes, cid_groups, parents)
        return loss.squeeze()

    @staticmethod
    def backward(ctx, grad_output):
        grad_prob = grad_label = grad_group_offsets = grad_group_sizes = grad_cid_groups = \
            grad_parents = grad_ignore_label = grad_axis = grad_valid_normalization = None

        if ctx.needs_input_grad[0]:
            ignore_label, axis = ctx.ignore_label, ctx.softmax_axis
            prob, label, norm, group_offsets, group_sizes, cid_groups, parents = ctx.saved_tensors
            has_ignore_label = ignore_label is not None
            grad_prob = smtl_cuda.backward(
                prob, label,
                parents, group_offsets, group_sizes, cid_groups,
                has_ignore_label, ignore_label,
                axis
            )[0] * grad_output / norm

        return (grad_prob, grad_label, grad_group_offsets, grad_group_sizes, grad_cid_groups,
                grad_parents, grad_ignore_label, grad_axis, grad_valid_normalization)


class SoftmaxTreeWithLoss(nn.Module):
    def __init__(self, tree, ignore_label=None, axis=1, loss_weight=1.0, valid_normalization=False):
        super(SoftmaxTreeWithLoss, self).__init__()
        self.tree = tree  # type: str
        self.ignore_label = ignore_label
        self.axis = axis
        self.loss_weight = loss_weight
        self.valid_normalization = valid_normalization
        if ignore_label is not None:
            assert ignore_label < 0, "Ignore label must be negative"

        group_offsets, group_sizes, cid_groups, parents, _, _, _ = read_softmax_tree(self.tree)
        self.register_buffer('group_offsets', torch.from_numpy(np.array(group_offsets, dtype=np.int32)))
        self.register_buffer('group_sizes', torch.from_numpy(np.array(group_sizes, dtype=np.int32)))
        self.register_buffer('cid_groups', torch.from_numpy(np.array(cid_groups, dtype=np.int32)))
        self.register_buffer('parents', torch.from_numpy(np.array(parents, dtype=np.int32)))
        self.node_count = len(cid_groups)
        self.group_count = len(group_offsets)

    def forward(self, x, label):
        return SoftmaxTreeWithLossFunction.apply(
            x, label,
            self.group_offsets, self.group_sizes, self.cid_groups, self.parents,
            self.ignore_label, self.axis, self.valid_normalization
        ) * self.loss_weight

    def extra_repr(self):
        """Extra information
        """
        return 'normalization={}, tree={}, nodes={}, groups={}{}{}{}'.format(
            "VALID" if self.valid_normalization else "BATCH_SIZE",
            self.tree, self.node_count, self.group_count,
            ", ignore_label={}".format(self.ignore_label) if self.ignore_label is not None else "",
            ", axis={}".format(self.axis) if self.axis != 1 else "",
            ", loss_weight={}".format(self.loss_weight) if self.loss_weight != 1.0 else ""
        )
