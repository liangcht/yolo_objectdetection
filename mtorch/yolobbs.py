import numpy as np
import torch
import torch.nn as nn


class YoloBBs(nn.Module):
    """Convert to Yolo bounding boxes
    """

    def __init__(self, biases=None, feat_stride=32):
        """YoloBBs
        :param biases: width/x then height/y anchor biases
        :param feat_stride: the feature stride for all the previous layers
        """
        super(YoloBBs, self).__init__()
        if biases is None:
            biases = []
        self.feat_stride = feat_stride
        self.num_anchor = len(biases) // 2
        assert self.num_anchor > 0, "Invalid number of biases"
        self.register_buffer('biases', torch.from_numpy(np.array(biases, dtype=np.float32).reshape(-1, 2)))

    def forward(self, xy, wh, im_info=None):
        """
        :param xy: torch tensor of b x 2a x h x w
        :param wh: torch tensor of b x 2a x h x w
        :param im_info: torch tensor of 2 (optional), width then height of image
        :returns torch tensor of b x a x h x w x 4
        """
        assert xy.size() == wh.size(), "xy and wh must have the same shape"
        assert xy.dim() == 4, "xy and wh must have 4 dimensions"
        n, anchors, height, width = xy.size()
        anchors //= 2
        assert self.num_anchor == anchors, "invalid number of anchors"
        bbs = xy.new_empty((n, anchors, height, width, 4))
        # tensor views into input
        x = xy[:, :anchors, :, :]
        y = xy[:, anchors:, :, :]
        w = wh[:, :anchors, :, :]
        h = wh[:, anchors:, :, :]

        # Note: avoid aliasing the output tensors, for AutoGrad (if we ever wanted to compute backward)

        # Use broadcasting to convert to Yolo bounding box in-place
        i = torch.arange(width, dtype=xy.dtype, device=xy.device)
        bbs[:, :, :, :, 0] = (x + i) / width
        del i
        j = torch.arange(height, dtype=xy.dtype, device=xy.device).view(-1, 1)
        bbs[:, :, :, :, 1] = (y + j) / height
        del j
        bbs[:, :, :, :, 2] = w.exp() * self.biases[:, 0].view(anchors, 1, 1) / width
        bbs[:, :, :, :, 3] = h.exp() * self.biases[:, 1].view(anchors, 1, 1) / height

        # Correct bounding boxes
        if im_info is None:
            return bbs
        assert im_info.numel() == 2, "im_info must have 2 values"
        net_h = self.feat_stride * height
        net_w = self.feat_stride * width
        im_h = im_info[0].item() or net_h
        im_w = im_info[1].item() or net_w

        if float(net_w) / im_w < float(net_h) / im_h:
            new_w = net_w
            new_h = (im_h * net_w) // im_w
        else:
            new_h = net_h
            new_w = (im_w * net_h) // im_h

        # Convert from network height and width to image height and width
        bbs[:, :, :, :, 0] = (bbs[:, :, :, :, 0] - (net_w - new_w) / 2. / net_w) / (float(new_w) / net_w)
        bbs[:, :, :, :, 1] = (bbs[:, :, :, :, 1] - (net_h - new_h) / 2. / net_h) / (float(new_h) / net_h)
        bbs[:, :, :, :, 2] *= float(net_w) / new_w
        bbs[:, :, :, :, 3] *= float(net_h) / new_h
        bbs[:, :, :, :, 0] *= im_w
        bbs[:, :, :, :, 2] *= im_w
        bbs[:, :, :, :, 1] *= im_h
        bbs[:, :, :, :, 3] *= im_h

        return bbs

    def extra_repr(self):
        """Extra information
        """
        return 'feat_stride={}, num_anchor={}'.format(
            self.feat_stride,
            self.num_anchor
        )


# TODO: make these a proper unit test
if __name__ == '__main__':
    net = YoloBBs(biases=[1.08, 1.19, 3.4, 4.4, 6, 11])

    xy_ = torch.empty(6, 2 * 3, 5, 8).uniform_()
    wh_ = torch.empty(6, 2 * 3, 5, 8).uniform_().sigmoid_()
    im_info_ = torch.Tensor([480, 640])
    b = net(xy_, wh_, im_info_)

    net = net.cuda()
    xy_ = xy_.cuda()
    wh_ = wh_.cuda()
    im_info_ = im_info_.cuda()
    b2 = net(xy_, wh_, im_info_)

    print((b2.cpu() - b).sum())
