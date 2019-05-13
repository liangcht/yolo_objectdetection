import torch
import torch.nn as nn

    
class YoloEvalCompat(nn.Module):
    """
    permutation of the channels to be the last 
    """
    def __init__(self):
        super(YoloEvalCompat, self).__init__()
        
    def forward(self, x):
        """
        :param x: torch tensor of b x c x a x h x w  or b x c x h x w
        """ 
        assert len(x.shape) >= 4, "invalid nuber of axises for x {}, expected >= {}".format(len(x.shape), 4)
        orig_order = list(range(x.dim()))
        new_order = [orig_order[0]] + orig_order[2:] + [orig_order[1]]
        return x.permute(new_order)


# TODO: make these a proper unit test
if __name__ == '__main__':

    net = YoloEvalCompat()

    # create a matrix 1x4x3x8 --> 1x3x8x4
    a = torch.rand(1, 4, 3, 8)
    b = net(a)
    print("Input:", a.shape)
    print("Output:", b.shape)
    
    # create a matrix 1x4x5x3x8 --> 1x5x3x8x4
    a = torch.rand(1, 4, 5, 3, 8)
    b = net(a)
    print("Input:", a.shape)
    print("Output:", b.shape)

    # create a matrix 1x4x5 --> Error
    a = torch.rand(1, 4, 5)
    b = net(a)
    print("Input:", a.shape)
    print("Output:", b.shape)
