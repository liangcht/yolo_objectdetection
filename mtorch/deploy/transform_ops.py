import torch.nn as nn


class FloatSubtract(nn.Module):
    def __init__(self, dc):
        """Subtract a constant from the tensor
        dc could be anything that can be subtracted,
            it could be per-pixel mean (like the landmark example)
            or per-channel mean (like the celeb) example.
        This is a simple subtraction, fewer operations means smaller network (no need to divide by 1.0 for example)
        There is also .float() which a trick so that we can pass bytes to the network and expand to larger float type
            later on in ONNX instead of in the managed code and save some cycles copying floats.
        :param dc: the constant to subtract
        """
        super(FloatSubtract, self).__init__()
        self.dc = dc

    def extra_repr(self):
        """Extra information
        """
        return 'dc.shape={}'.format(self.dc.shape)

    def forward(self, x):
        return x.float() - self.dc.to(x.device)


class CenterCrop(nn.Module):
    def __init__(self, crop_size):
        """Crop from the center of a 4d tensor
        :param crop_size: the center crop size
        """
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def extra_repr(self):
        """Extra information
        """
        return 'crop_size={}'.format(
            self.crop_size
        )

    def forward(self, x):
        assert x.shape[2] == x.shape[3] and x.shape[2] > self.crop_size + 2
        offset = (x.shape[2] - self.crop_size) / 2
        return x[:, :, offset:offset + self.crop_size, offset:offset + self.crop_size]
