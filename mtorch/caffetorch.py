import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        n_b = x.data.size(0)
        x = x.view(n_b, -1)
        return x

    def __repr__(self):
        return 'view(nB, -1)'


class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1, len(inputs)):
                x = torch.max(x, inputs[i])
        else:
            raise RuntimeError('forward Eltwise, unknown operator: {}'.format(self.operation))
        return x


class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels

    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
        n_b = x.size(0)
        n_c = x.size(1)
        n_h = x.size(2)
        n_w = x.size(3)
        x = x * self.weight.view(1, n_c, 1, 1).expand(n_b, n_c, n_h, n_w) + \
            self.bias.view(1, n_c, 1, 1).expand(n_b, n_c, n_h, n_w)
        return x


class Crop(nn.Module):
    def __init__(self, axis, offset):
        super(Crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def __repr__(self):
        return 'Crop(axis=%d, offset=%d)' % (self.axis, self.offset)

    def forward(self, x, ref):
        for axis in range(self.axis, x.dim()):
            ref_size = ref.size(axis)
            indices = torch.arange(self.offset, self.offset + ref_size).long()
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            x = x.index_select(axis, Variable(indices))
        return x


class Slice(nn.Module):
    def __init__(self, axis, slice_points):
        super(Slice, self).__init__()
        self.axis = axis
        self.slice_points = slice_points

    def __repr__(self):
        return 'Slice(axis=%d, slice_points=%s)' % (self.axis, self.slice_points)

    def forward(self, x):
        prev = 0
        outputs = []
        is_cuda = x.data.is_cuda
        device_id = None
        if is_cuda:
            device_id = x.data.get_device()
        for idx, slice_point in enumerate(self.slice_points):
            rng = range(prev, slice_point)
            rng = torch.LongTensor(rng)
            if is_cuda:
                rng = rng.cuda(device_id)
            rng = Variable(rng)
            y = x.index_select(self.axis, rng)
            prev = slice_point
            outputs.append(y)
        return tuple(outputs)


class Concat(nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Concat(axis=%d)' % self.axis

    def forward(self, *inputs):
        return torch.cat(inputs, self.axis)


class Permute(nn.Module):
    def __init__(self, order0, order1, order2, order3):
        super(Permute, self).__init__()
        self.order0 = order0
        self.order1 = order1
        self.order2 = order2
        self.order3 = order3

    def __repr__(self):
        return 'Permute(%d, %d, %d, %d)' % (self.order0, self.order1, self.order2, self.order3)

    def forward(self, x):
        x = x.permute(self.order0, self.order1, self.order2, self.order3).contiguous()
        return x


class EuclideanLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EuclideanLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, output, target, weights=None):
        diff = (output - target).pow(2)
        if weights is None:
            return diff.sum() / (2 * output.shape[0]) * self.loss_weight
        return diff.view(-1).dot(weights.view(-1)) / (2 * output.shape[0]) * self.loss_weight

    def extra_repr(self):
        """Extra information
        """
        return '{}'.format(
            "loss_weight={}".format(self.loss_weight) if self.loss_weight != 1.0 else ""
        )


class SoftmaxWithLoss(nn.CrossEntropyLoss):
    def __init__(self, loss_weight=1.0, ignore_label=None, valid_normalization=False):
        """SoftmaxWithLoss
        :param loss_weight: loss weight to multiply to
        :param ignore_label: the label to ignore when calculating
        :param valid_normalization: if should normalzie by total non-ignored labels
        """
        super(SoftmaxWithLoss, self).__init__(
            ignore_index=ignore_label if ignore_label is not None else -100,
            size_average=valid_normalization
        )
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.valid_normalization = valid_normalization

    def forward(self, x, targets):
        targets = targets.long()
        loss = nn.CrossEntropyLoss.forward(self, x, targets) * self.loss_weight
        if self.valid_normalization:
            return loss
        # loss is sum, normalize by BATCH
        return loss / x.size(0)

    def extra_repr(self):
        """Extra information
        """
        return 'normalization={}{}{}'.format(
            "VALID" if self.valid_normalization else "BATCH_SIZE",
            ", loss_weight={}".format(self.loss_weight) if self.loss_weight != 1.0 else "",
            ", ignore_label={}".format(self.ignore_label) if self.ignore_label is not None else "",
        )


class Normalize(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super(Normalize, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def __repr__(self):
        return 'Normalize(channels=%d, scale=%f)' % (self.n_channels, self.scale)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class Flatten(nn.Module):
    def __init__(self, axis):
        super(Flatten, self).__init__()
        self.axis = axis

    def __repr__(self):
        return 'Flatten(axis=%d)' % self.axis

    def forward(self, x):
        left_size = 1
        for i in range(self.axis):
            left_size = x.size(i) * left_size
        return x.view(left_size, -1).contiguous()


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __repr__(self):
        return 'Accuracy()'

    # noinspection PyMethodMayBeStatic
    def forward(self, output, label):
        max_vals, max_ids = output.data.max(1)
        # noinspection PyUnresolvedReferences
        n_correct = (max_ids.view(-1).float() == label.data).sum()
        batchsize = output.data.size(0)
        accuracy = float(n_correct) / batchsize
        print('accuracy = %f', accuracy)
        accuracy = output.data.new().resize_(1).fill_(accuracy)
        return Variable(accuracy)


class Reorg(nn.Module):
    """Reorganize data blob to reduce spatial resolution by increasing channels, or vice versa
    """
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def __repr__(self):
        return 'Reorg(stride={})'.format(self.stride)

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        n = x.data.size(0)
        c = x.data.size(1)
        h = x.data.size(2)
        w = x.data.size(3)
        assert (h % stride == 0)
        assert (w % stride == 0)
        ws = stride
        hs = stride
        x = x.view(n, c, h // hs, hs, w // ws, ws).transpose(3, 4).contiguous()
        x = x.view(n, c, h // hs * w // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(n, c, hs * ws, h // hs, w // ws).transpose(1, 2).contiguous()
        x = x.view(n, hs * ws * c, h // hs, w // ws)
        return x
