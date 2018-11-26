import numpy as np
import torch.nn as nn


def reshape_helper(orig_dims, reshape_dims, axis=0, num_axes=-1):
    if num_axes == -1:
        num_axes = len(orig_dims[axis:])
    end_axis = axis + num_axes
    new_dims = list(orig_dims[:axis])
    count = np.prod(orig_dims[axis:end_axis])
    cur = 1
    for idx, d in enumerate(reshape_dims):
        if d == 0:
            d = orig_dims[axis + idx]
        elif d < 0:
            d = int(count / cur)
        new_dims.append(d)
        cur *= d
    new_dims += list(orig_dims[end_axis:])
    assert np.prod(orig_dims) == np.prod(new_dims), "Reshape: shape count: {} != {}".format(orig_dims, new_dims)

    return new_dims


class Reshape(nn.Module):
    def __init__(self, dims, axis, num_axes):
        super(Reshape, self).__init__()
        self.dims = dims
        self.axis = axis
        self.num_axes = num_axes

    def extra_repr(self):
        """Extra information
        """
        return 'dims={}{}{}'.format(
            self.dims,
            ", axis={}".format(self.axis) if self.axis else "",
            ", num_axes={}".format(self.num_axes) if self.num_axes >= 0 else "",
        )

    def forward(self, x):
        new_dims = reshape_helper(x.shape, self.dims, axis=self.axis, num_axes=self.num_axes)
        return x.view(*new_dims).contiguous()
