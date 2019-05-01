import torch
import torch.nn as nn


class CCSLoss(nn.Module):
    """ Classification vector-centered Cosine Similarity (CCS) loss at https://arxiv.org/pdf/1707.05574.pdf
    """
    def __init__(self):
        super(CCSLoss, self).__init__()

    def forward(self, feature, weight, label):
        """
        :param feature: tensor of size N*D, where N is the batch size, D is the dimension
        :param weight: tensor of size C*D, where C is the number of classes,
        :param label: tensor of size N, each value is int in [0, C)
        """

        # no backward for weight
        weight = weight.detach()
        label = label.detach()
        num_samples, fea_dim = feature.shape
        num_cls, w_dim = weight.shape
        assert fea_dim == w_dim, "feature dim {} does not match with weight dim {}".format(fea_dim, w_dim)
        assert num_samples == label.shape[0]

        new_weight = weight[label]
        dot_product = feature * new_weight
        dot_product = torch.sum(dot_product, dim=1)
        fea_norm = torch.norm(feature, dim=1)
        w_norm = torch.norm(new_weight, dim=1)
        loss = -1 * torch.mean(dot_product / (fea_norm * w_norm))

        return loss

