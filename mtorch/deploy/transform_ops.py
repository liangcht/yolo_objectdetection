import torch
import torch.nn as nn


class ToFloat(nn.Module):
    def __init__(self):
        """Convert to .float()
        """
        super(ToFloat, self).__init__()

    def forward(self, x):
        return x.float()


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


@torch.jit.script
def center_slice_helper(x, h_offset, w_offset, h_end, w_end):
    return x[:, :, h_offset:h_end, w_offset:w_end]


class CenterCrop(nn.Module):
    def __init__(self, crop_size):
        """Crop from the center of a 4d tensor
        Input shape can be dynamic
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
        h_offset = (x.shape[2] - self.crop_size) / 2
        w_offset = (x.shape[3] - self.crop_size) / 2
        h_end = h_offset + self.crop_size
        w_end = w_offset + self.crop_size
        return center_slice_helper(x, h_offset, w_offset, h_end, w_end)


class CenterCropFixed(nn.Module):
    def __init__(self, crop_size):
        """Crop from the center of a 4d tensor
        Use this only if the input image size is always fixed
        :param crop_size: the center crop size
        """
        super(CenterCropFixed, self).__init__()
        self.crop_size = crop_size

    def extra_repr(self):
        """Extra information
        """
        return 'crop_size={}'.format(
            self.crop_size
        )

    def forward(self, x):
        h_offset = (x.shape[2] - self.crop_size) / 2
        w_offset = (x.shape[3] - self.crop_size) / 2
        return x[:, :, h_offset:h_offset + self.crop_size, w_offset:w_offset + self.crop_size]
