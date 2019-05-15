import torch
from PIL import Image

from mmod.detection import result2bblist
from mmod.simple_parser import load_labelmap_list
from mtorch.augmentation import TestAugmentation
from mtorch.darknet import darknet_layers
from mtorch.yolo_predict import (PlainPredictorClassSpecificNMS,
                                 TreePredictorClassSpecificNMS)
from mtorch.yolo_v2 import (yolo_0extraconv, yolo_1extraconv, yolo_2extraconv,
                            yolo_3extraconv)


def load_model(path_model, num_classes, num_extra_convs, is_caffemodel):
    """Creates a yolo model for evaluation
    :param path_model, str, path to latest checkpoint
    :param num_classes: int, number of classes to detect
    :param is_caffemodel, bool, if true, assumes model weights are derived from caffemodel
    :return model: nn.Sequential or nn.Module
    """
    yolo = [yolo_0extraconv, yolo_1extraconv, yolo_2extraconv, yolo_3extraconv][num_extra_convs]

    model = yolo(darknet_layers(), 
             weights_file=path_model,
             caffe_format_weights=is_caffemodel,
             num_classes=num_classes)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    return model


def get_predictor(num_classes, tree):
    """Creates a yolo model for evaluation
    :param num_classes, int, number of classes to detect
    :param tree, str, path to a tree structure, or None
    :return model: nn.Sequential or nn.Module
    """
    if tree:
        predictor = TreePredictorClassSpecificNMS(tree, num_classes=num_classes)
    else:
        predictor = PlainPredictorClassSpecificNMS(num_classes=num_classes)
    
    if torch.cuda.is_available():
        predictor = predictor.cuda()

    return predictor    

def prepare_image(image, transform=None):
    """Convert the image to the required format and apply necessary transforms to the image
    :param image, numpy arr in BGR format, image to be transformed
    :param transform, augmentations to perform, if any
    :return transformed image
    """
    image = image[:, :, ::-1] # BGR to RGB
    image = Image.fromarray(image.astype('uint8'), mode='RGB')  # save in PIL format

    w, h = image.size

    if transform:
        image = transform(image)

    return image, h, w


class YoloDetector(object):
    """Class for Yolo predictions
    Parameters:
        path_model: str, path to latest checkpoint
        path_labelmap: str, path to labelmap
        thresh: float, confidence threshold for final prediction, default 0
        obj_threshold: float, objectness threshold for final prediction, default 0
        path_tree: str, path to a tree structure, it prediction based on tree is required, if any
        num_extra_convs: int, number of extra conv used by the yolo model, default 2
        is_caffemodel, bool, if true, assumes model weights are derived from caffemodel, default False
    """

    def __init__(self, path_model, path_labelmap, thresh=0.0, obj_thresh=0.0, path_tree=None, num_extra_convs=2, is_caffemodel=False):
        self.cmap = load_labelmap_list(path_labelmap)
        self.model = load_model(path_model=path_model, num_classes=len(self.cmap), num_extra_convs=num_extra_convs, is_caffemodel=is_caffemodel)
        self.predictor = get_predictor(num_classes=len(self.cmap), tree=path_tree)

        self.thresh = thresh
        self.obj_thresh = obj_thresh

        self.transform = TestAugmentation()


    def detect(self, image):
        """Returns the YOLO predictions on the given image
        :param: image, numpy arr in BGR format, input image
        :return predictions of the yolo network
        """
        image, h, w = prepare_image(image, self.transform())
        image = image.unsqueeze_(0)
        image = image.float()
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            features = self.model(image)
        prob, bbox = self.predictor(features, torch.Tensor((h, w)))

        bbox = bbox.cpu().numpy()
        prob = prob.cpu().numpy()


        assert bbox.shape[-1] == 4
        bbox = bbox.reshape(-1, 4)
        prob = prob.reshape(-1, prob.shape[-1])
        result = result2bblist((h, w), prob, bbox, self.cmap,
                                thresh=self.thresh, obj_thresh=self.obj_thresh)
        return result
