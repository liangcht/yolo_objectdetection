import torch


def correct_region_boxes(boxes, im_w, im_h, net_w, net_h, relative=False):
    """
    :param boxes: [N, 4], x, y, w, h in relative coords
    """
    out_boxes = boxes.clone()

    if net_w / im_w < net_h / im_h:
        new_w = net_w
        new_h = (im_h * net_w) / im_w
    else:
        new_w = (im_w * net_h) / im_h
        new_h = net_h

    w_off = (net_w - new_w) / 2. / net_w
    h_off = (net_h - new_h) / 2. / net_h
    w_scale = net_w / new_w
    h_scale = net_h / new_h
    out_boxes[:, 0] = (boxes[:, 0] - w_off) * w_scale
    out_boxes[:, 1] = (boxes[:, 1] - h_off) * h_scale
    out_boxes[:, 2] = boxes[:, 2] * w_scale
    out_boxes[:, 3] = boxes[:, 3] * h_scale

    if not relative:
        out_boxes[:, [0, 2]] *= im_w
        out_boxes[:, [1, 3]] *= im_h

    return out_boxes


def clip_boxes_in_scope(boxes, im_w, im_h, probs=None):
    # clip into image scope
    out_boxes = boxes.clone()
    min_x, max_x = boxes.new(1).fill_(0), boxes.new(1).fill_(im_w - 1)
    min_y, max_y = boxes.new(1).fill_(0), boxes.new(1).fill_(im_h - 1)
    x1 = out_boxes[:, 0] - out_boxes[:, 2] / 2
    y1 = out_boxes[:, 1] - out_boxes[:, 3] / 2
    x2 = out_boxes[:, 0] + out_boxes[:, 2] / 2
    y2 = out_boxes[:, 1] + out_boxes[:, 3] / 2
    x1 = torch.min(torch.max(x1, min_x), max_x)
    y1 = torch.min(torch.max(y1, min_y), max_y)
    x2 = torch.min(torch.max(x2, min_x), max_x)
    y2 = torch.min(torch.max(y2, min_y), max_y)

    out_boxes[:, 0] = (x1 + x2) / 2
    out_boxes[:, 1] = (y1 + y2) / 2
    out_boxes[:, 2] = x2 - x1
    out_boxes[:, 3] = y2 - y1

    valid_idxs = torch.nonzero(out_boxes[:, 2] * out_boxes[:, 3] > 0).squeeze()
    out_boxes = out_boxes.index_select(0, valid_idxs)
    if probs is None:
        return out_boxes  # xywh
    else:
        return out_boxes, probs.index_select(0, valid_idxs)


def xywh2corner(boxes):
    """
    convert [x,y,w,h] to [x1,y1,x2,y2]
    :param boxes:
    :return:
    """
    if boxes.numel() == 0:
        return torch.tensor([])

    x, y = boxes[:, 0], boxes[:, 1]
    w, h = boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=1)


def filter_box_by_prob(boxes, probs):
    if isinstance(boxes, list):
        assert isinstance(probs, list), 'probs should match the type of boxes [list]'
        filtered_boxes = []
        for box, prob in zip(boxes, probs):
            filtered_boxes.append(filter_box_by_prob(box, prob))
    else:
        assert boxes.size(0) == probs.size(0)
        max_cls_prob, _ = torch.max(probs[:, :-1], dim=1, keepdim=True)
        keep_idxs = torch.nonzero(max_cls_prob > 0.5).squeeze()
        return boxes.index_select(0, keep_idxs)
