import torch
import torch.nn as nn

from mtorch.regionloss_utils  import region_util
from mtorch.regionloss_utils.cython_bbox import bbox_ious, bbox_ious_diag, anchor_intersections
from mtorch.caffetorch import EuclideanLoss

DEFAULT_ANCHORS = torch.Tensor([[1.08, 1.19],
                               [3.42, 4.41],
                               [6.63, 11.38],
                               [9.42, 5.11],
                               [16.62, 10.52]])


class RegionLoss(nn.Module):
    """ 
    This class is reproducing RegionLoss of Caffe in Pytho. 
    NOTE: it reshapes the input similarly to RegiontTarget of Caffe and Pytorch
    """
    # TODO: add documentation of arguments
   
    def __init__(self, classes=20, anchors=DEFAULT_ANCHORS, coords=4,
                 obj_esc_thresh=0.6, rescore=True, object_scale=5.0,
                 class_scale=1.0, noobject_scale=1.0, coord_scale=1.0,
                 anchor_aligned_images=12800, ngpu=1):
        super(RegionLoss, self).__init__()

        self.classes = classes
        self.register_buffer('anchors', anchors)
        self.coords = coords
        self.anchor_num = self.anchors.size(0)
        self.channel = self.anchor_num * (coords + classes + 1)
        self.channel_per_anchor = coords + classes + 1
        self.seen = 0
        self.obj_esc_thresh = obj_esc_thresh
        self.rescore = rescore
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.noobject_scale = noobject_scale
        self.coord_scale = coord_scale
        self.anchor_aligned_images = anchor_aligned_images
        self.ngpu = ngpu
        self.sigmoid = nn.Sigmoid()
        self.weighted_MSE = EuclideanLoss()
        self.cross_ent  = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        assert coords == 4, 'only support coords==4 now'

    def forward(self, x, label):
        """
        :param x: torch tensor, the input to the loss layer
        :param target: [N x (coords+1)], expected format: x,y,w,h,cls
        :return: float32 loss scalar 
        """
        batch, out_h, out_w = x.size(0), x.size(2), x.size(3)
        target = label.view(batch, -1, self.anchor_num).contiguous()
        assert x.size(1) == self.channel, \
            'channel should be %d, but get %d' % (self.channel, x.size(1))

        x_res = x.view(batch, -1, self.anchor_num, out_h, out_w).contiguous()
        x_res = x_res.permute(0, 2, 1, 3, 4)
        # tx, ty, tw, th, to -> sig(tx), sig(ty), exp(tw), exp(th), sig(to)
        x = self.sigmoid(x_res[:, :, 0, :, :])
        y = self.sigmoid(x_res[:, :, 1, :, :])
        w = x_res[:, :, 2, :, :]
        h = x_res[:, :, 3, :, :]
    
        grid_x = torch.arange(0, out_h * out_w) % out_w
        grid_y = torch.floor(torch.arange(0, out_h * out_w).float() / out_w)
        grid_x = grid_x.view(1, 1, out_h, out_w).type_as(x.data)
        grid_y = grid_y.view(1, 1, out_h, out_w).type_as(y.data)

        anchor_w = self.anchors[:, 0].contiguous().view(1, self.anchor_num, 1, 1).type_as(w.data)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.anchor_num, 1, 1).type_as(h.data)

        pred_x = grid_x + x.data
        pred_y = grid_y + y.data
        pred_w = torch.exp(w.data)*anchor_w
        pred_h = torch.exp(h.data)*anchor_h
        pred_boxes = torch.stack((pred_x, pred_y, pred_w, pred_h), dim=4)
        pred_boxes = pred_boxes.view(batch, -1, 4)

        conf = self.sigmoid(x_res[:, :, 4, :, :])
        clas_prob = x_res[:, :, 5:, :, :]
        org = clas_prob.permute(0, 2, 1, 3, 4).view(batch, -1 , out_h, out_w)
        clas_prob = org.view(batch, self.anchor_num, -1, out_h, out_w). permute(0, 2, 1, 3, 4)
        coord_mask, conf_mask, tx, ty, tw, th, tconf, tcls = \
            self.build_targets(pred_boxes, target.data, out_h, out_w)

        self.loss_xy = self.weighted_MSE(x, tx, coord_mask) + self.weighted_MSE(y, ty, coord_mask)
        self.loss_wh = self.weighted_MSE(w, tw, coord_mask) + self.weighted_MSE(h, th, coord_mask)
        self.loss_conf = self.weighted_MSE(conf, tconf, conf_mask)
        self.loss_cls = self.cross_ent(clas_prob, tcls) / batch 
        self.loss = self.loss_xy + self.loss_wh + self.loss_conf + self.loss_cls

        self.seen += batch * self.ngpu
        return self.loss

    def build_targets(self, pred_boxes, target, nH, nW):
        """ 
        Given the predictions and taget prepares individual componnets required for loss calculation
        :param pred_boxes: predictions of bounding boxes
        :param target: groupnd truth
        :param nH: height of perdiction tensor
        :param nW: width of prediction tensor 
        :return coord_mask: which coordinates should participate in the xywh loss
        :return conf_mask: which cooridnates should participate in objectness loss
        :return tx: float32, target x
        :return ty: float32, target y
        :return tw: float32, target w
        :return th: float32, target h
        :return tconf: float32, target objectnes
        :return tcls: uint64 expected classes
        """
        nB = target.size(0)
        anchor_step = self.anchors.size(1)
        conf_mask = torch.ones(nB, self.anchor_num, nH, nW).type_as(pred_boxes) * self.noobject_scale
        coord_mask = torch.zeros(nB, self.anchor_num, nH, nW).type_as(pred_boxes)
        tx = torch.zeros(nB, self.anchor_num, nH, nW).type_as(pred_boxes)
        ty = torch.zeros(nB, self.anchor_num, nH, nW).type_as(pred_boxes)
        tw = torch.zeros(nB, self.anchor_num, nH, nW).type_as(pred_boxes)
        th = torch.zeros(nB, self.anchor_num, nH, nW).type_as(pred_boxes)
        tconf = torch.zeros(nB, self.anchor_num, nH, nW).type_as(pred_boxes)
        l_type = 'torch.cuda.LongTensor' if pred_boxes.is_cuda else 'torch.LongTensor'
        tcls = torch.zeros(nB, self.anchor_num, nH, nW).type(l_type)
        tcls.fill_(-1)
        # for each pred box, find the best overlapped gt box,
        # ignore the penalty for some pred boxes
        for b in range(nB):
            curr_pred_boxes = pred_boxes[b]
            valid_gt_idxs = torch.nonzero(target[b, :, 0] != 0).squeeze()
            if valid_gt_idxs.nelement() == 0:
                continue
            valid_gt_boxes = target[b, :, :self.coords].index_select(0, valid_gt_idxs)
            valid_gt_boxes[:, [0, 2]] *= nW
            valid_gt_boxes[:, [1, 3]] *= nH
            curr_ious = bbox_ious(region_util.xywh2corner(curr_pred_boxes).cpu().numpy(),
                                  region_util.xywh2corner(valid_gt_boxes).cpu().numpy()).max(axis=1)
            curr_ious = torch.from_numpy(curr_ious).view(self.anchor_num, nH, nW).type_as(pred_boxes)
            conf_mask[b][curr_ious > self.obj_esc_thresh] = 0

        if self.seen < self.anchor_aligned_images:
            if anchor_step == 4:
                tx = self.anchors.index_select(1, torch.LongTensor([2])) \
                    .view(1, self.anchor_num, 1, 1).repeat(nB, 1, nH, nW)
                ty = self.anchors.index_select(1, torch.LongTensor([3])) \
                    .view(1, self.anchor_num, 1, 1).repeat(nB, 1, nH, nW)
            else:
                tx.fill_(0.5)
                ty.fill_(0.5)
            tw.zero_()
            th.zero_()
            coord_mask.fill_(0.01)

        # for each gt box, find the best overlapped anchor
        for b in range(nB):
            valid_gt_idxs = torch.nonzero(target[b, :, 0]).squeeze()
            valid_gt_clses = target[b, :, self.coords].index_select(0, valid_gt_idxs).type(l_type)
            valid_gt_boxes = target[b, :, :self.coords].index_select(0, valid_gt_idxs)
            try:
                valid_gt_grids = valid_gt_boxes[:, [0, 1]]
            except Exception as err:
                continue
            valid_gt_grids[:, 0] *= nW
            valid_gt_grids[:, 1] *= nH
            valid_gt_grids = valid_gt_grids.type(l_type)

            shift_gt_boxes = valid_gt_boxes.clone()
            shift_gt_boxes[:, [0, 1]] = 0
            shift_gt_boxes[:, 2] *= nW
            shift_gt_boxes[:, 3] *= nH
            curr_ious, max_anchor_idxs = torch.max(torch.from_numpy(
                                                   anchor_intersections(self.anchors.cpu().numpy(),
                                                                        shift_gt_boxes.cpu().numpy()
                                                                        )), dim=0)
            max_anchor_idxs = max_anchor_idxs.type(l_type)
            gt_tx = valid_gt_boxes[:, 0]*nW - valid_gt_grids[:, 0].clone().type_as(valid_gt_boxes)
            gt_ty = valid_gt_boxes[:, 1]*nH - valid_gt_grids[:, 1].clone().type_as(valid_gt_boxes)
            gt_tw = torch.log(valid_gt_boxes[:, 2]*nW / self.anchors.index_select(0, max_anchor_idxs)[:, 0])
            gt_th = torch.log(valid_gt_boxes[:, 3]*nH / self.anchors.index_select(0, max_anchor_idxs)[:, 1])

            coord_mask[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = self.coord_scale * (
                        2 - valid_gt_boxes[:, 2] * valid_gt_boxes[:, 3])
            conf_mask[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = self.object_scale
            tx[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = gt_tx
            ty[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = gt_ty
            tw[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = gt_tw
            th[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = gt_th

            if self.rescore:
                t_pred_idxs = max_anchor_idxs * nH * nW + valid_gt_grids[:, 1] * nW + valid_gt_grids[:, 0]
                t_pred_boxes = pred_boxes[b].index_select(0, t_pred_idxs)
                # valid_gt_boxes is still relative coordinate
                t_pred_boxes[:, [0, 2]] /= nW
                t_pred_boxes[:, [1, 3]] /= nH
                curr_ious = torch.from_numpy(bbox_ious_diag(region_util.xywh2corner(valid_gt_boxes).cpu().numpy(),
                                             region_util.xywh2corner(t_pred_boxes).cpu().numpy())).type_as(tconf)  # best_iou
                tconf[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = curr_ious
            else:
                tconf[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] = 1

            tcls[b][(max_anchor_idxs, valid_gt_grids[:, 1], valid_gt_grids[:, 0])] =  valid_gt_clses

        return coord_mask, conf_mask, tx, ty, tw, th, tconf, tcls




