import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
from PIL import Image
from torchvision.transforms.functional import crop

from mmod.simple_parser import load_labelmap_list
from mtorch.augmentation import ClassifierTestAugmentation
from mtorch.tbox_utils import region_crop


def load_model(path_model, path_labelmap):
    """Creates a classifier model for evaluation
    :param path_model, str, path to latest checkpoint in *.pth.tar format
    :param path_labelmap: str, path to labelmap
    """
    checkpoint = torch.load(path_model)
    arch = checkpoint['arch']
    model = models.__dict__[arch](num_classes=checkpoint['num_classes'])
    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})".format(path_model, checkpoint['epoch']))

    # switch to evaluate mode
    model.eval()

    # load labelmap
    if path_labelmap:
        labelmap = load_labelmap_list(path_labelmap)
    elif 'labelmap' in checkpoint:
        labelmap = model['labelmap']
    else:
        labelmap = [str(i) for i in range(checkpoint['num_classes'])]

    return model, labelmap


def prepare_image(image, rect=None, transform=None):
    """Convert the image to the required format and apply necessary transforms to the image
    :param image, numpy arr in BGR format, image to be transformed
    :param rect, list of coordinates [x1, y1, x2, y2] if need to crop
    :param transform, augmentations to perform, if any
    :return transformed image
    """
    image = image[:, :, (2, 1, 0)]  # BGR to RGB
    image = Image.fromarray(image, mode='RGB')  # save in PIL format

    if rect:
        crop_box = [float(val) for val in rect]
        image = region_crop(image, crop_box)

    if transform:
        image = transform(image)

    return image


class ImageClassifier(object):
    """Class for Classifier predictions
        Parameters:
        path_model: str, path to latest checkpoint in *.pth.tar format
        path_labelmap: str, path to labelmap
    """

    def __init__(self, path_model, path_labelmap):
        self.model, self.labelmap = load_model(path_model, path_labelmap)
        self.transform = ClassifierTestAugmentation()


    def predict(self, image, rect=None, topk=1):
        """Returns the classifier's predictions on the given image
        :param: image, numpy arr in BGR format, input image (in BGR format)
        :rect, list of coordinates [x1, y1, x2, y2] if need to crop
        :topk, int, top k result (default: 1)
        :return predictions of the yolo network
        """
        image = prepare_image(image, rect, self.transform())
        image = image.unsqueeze_(0)
        image = image.float()
        if torch.cuda.is_available():
            image = image.cuda()

        # compute output
        output = self.model(image)
        output = output.cpu()

        _, pred_topk = output.topk(topk, dim=1, largest=True)

        results = []
        n = 0
        for k in range(pred_topk.shape[1]):
            pred = {"class": self.labelmap[pred_topk[n, k]], "conf":  output[n, pred_topk[n, k]].item()}
            results.append(pred)
        
        return results
