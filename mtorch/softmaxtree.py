import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

from mmod.simple_parser import read_softmax_tree

import smt_cuda


class SoftmaxTreeFunction(Function):
    @staticmethod
    def forward(ctx, x, group_offsets, group_sizes, axis):
        assert 0 <= axis < x.dim(), "invalid axis for x of size: {}".format(x.size())
        node_count = group_offsets[-1] + group_sizes[-1]
        assert x.size(axis) == node_count, "Channel count: {} must match tree node count: {}".format(
            x.size(axis), node_count
        )
        prob = smt_cuda.forward(x, group_offsets, group_sizes, axis)[0]
        ctx.save_for_backward(prob, group_offsets, group_sizes, axis)
        return prob

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_group_offsets = grad_group_sizes = grad_axis = None
        if ctx.needs_input_grad[0]:
            prob, group_offsets, group_sizes, axis = ctx.saved_tensors
            grad_x = smt_cuda.backward(
                prob, grad_output,
                group_offsets, group_sizes,
                axis
            )[0]

        return grad_x, grad_group_offsets, grad_group_sizes, grad_axis


class SoftmaxTree(nn.Module):
    def __init__(self, tree, axis=1):
        super(SoftmaxTree, self).__init__()
        self.tree = tree  # type: str
        self.axis = axis

        group_offsets, group_sizes, cid_groups, parents = read_softmax_tree(self.tree)
        self.register_buffer('group_offsets', torch.from_numpy(np.array(group_offsets, dtype=np.int32)))
        self.register_buffer('group_sizes', torch.from_numpy(np.array(group_sizes, dtype=np.int32)))
        self.node_count = len(cid_groups)
        self.group_count = len(group_offsets)
        assert self.node_count == group_offsets[-1] + group_sizes[-1], "node count: {} last group: {}+{}".format(
            self.node_count, group_offsets[-1], group_sizes[-1]
        )

    def forward(self, x):
        return SoftmaxTreeFunction.apply(
            x,
            self.group_offsets, self.group_sizes, self.axis
        )

    def extra_repr(self):
        """Extra information
        """
        return 'tree={}, nodes={}, groups={}{}'.format(
            self.tree, self.node_count, self.group_count, ", axis={}".format(self.axis) if self.axis != 1 else ""
        )


if __name__ == '__main__':
    from StringIO import StringIO

    net = SoftmaxTree(StringIO("boz -1\nbozak -1\ngoat -1\n")).cuda()
    a = torch.rand(4, 3, 8).cuda()
    b = net(a)
    print(b[0, :, 0].sum())
