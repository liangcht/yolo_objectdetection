import random
import torch
import torchvision
from PIL import Image
import numpy as np

from mtorch.predict_transforms import ODImResize


def _adjust_bboxes(bboxes):
    new_bboxes = []
    for bbox in bboxes:
        center_x = (bbox[1] + bbox[3]) / 2
        center_y = (bbox[2] + bbox[4]) / 2
        if center_x >= 0 and center_x <= 1 and center_y >= 0 and center_y <= 1 and bbox[1] < bbox[3] and bbox[2] < bbox[4]:
            new_bboxes.append([bbox[0], max(bbox[1], 0.0), max(bbox[2], 0.0), min(bbox[3], 1.0), min(bbox[4], 1.0)])

    return new_bboxes

class ODImageTransform(object):
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img, target):
        img = self.transform(img)
        return img, target
    
class ODResize(object):
    """ Resize so that the shorter side will be self.size. """
    def __init__(self, size):
        self.size = size
        self.interpolation = Image.BILINEAR
        
    def __call__(self, img, target):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, target
        ow = self.size
        oh = self.size
        if w < h:
            oh = self.size * h // w
        else:
            ow = self.size * w // h
        
        img = img.resize((ow, oh), self.interpolation)           
        return img, target

class ODCenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, img, target):
        w, h = img.size
        x = int(round((w - self.output_size) / 2))
        y = int(round((h - self.output_size) / 2))
        img = img.crop((x, y, x + self.output_size, y + self.output_size))

        relative_x = x / w
        relative_y = y / h
        relative_w = self.output_size / w
        relative_h = self.output_size / h
        
        for i, t in enumerate(target):
            target[i] = (t[0], (t[1] - relative_x) / relative_w, (t[2] - relative_y) / relative_h,
                         (t[3] - relative_x) / relative_w, (t[4] - relative_y) / relative_h)

        target = _adjust_bboxes(target)
            
        return img, target

class ODRandomExpand(object):
    def __init__(self, background_color = (0,0,0)):
        self.background_color = (int(background_color[0] * 255), int(background_color[1] * 255), int(background_color[2] * 255))
        
    def __call__(self, img, target):
        if random.random() < 0.5:
            w, h = img.size
            ratio = random.uniform(1, 4)
            left = random.randint(0, int(w * ratio) - w)
            top = random.randint(0, int(h * ratio) - h)
            new_image = Image.new(img.mode, (int(w * ratio), int(h * ratio)), self.background_color)
            new_image.paste(img, (left, top))
            img = new_image

            relative_left = left / int(w * ratio)
            relative_top = top / int(h * ratio)
            
            for i, t in enumerate(target):
                target[i] = (t[0], relative_left + t[1] / ratio, relative_top + t[2] / ratio,
                             relative_left + t[3] / ratio, relative_top + t[4] / ratio)
                
        return img, target
    
class ODRandomHorizontalFlip(object):
    def __call__(self, img, target):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for i, t in enumerate(target):
                target[i] = (t[0], 1.0 - t[3], t[2], 1.0 - t[1], t[4])
        return img, target

class ODRandomResizedCrop(object):
    def __init__(self, input_size, overlaps=(0.1, 0.3, 0.5, 0.7, 0.9)):
        self.input_size = input_size
        self.interpolation = Image.BILINEAR
        self.scale_range = (0.1, 1)
        self.aspect_ratio_range = (0.5, 2)
        self.overlaps = overlaps
        
    def __call__(self, img, target):
        mode = random.randint(0,2)
        if mode == 0: # Use the entire original input image
            img, target = ODResize(self.input_size)(img, target)
            img, target = ODCenterCrop(self.input_size)(img, target)
            return img, target
        elif mode == 1: # Sample a patch so that the minimum iou with the objects is 0.1, 0.3, ...
            min_overlap = random.choice(self.overlaps)
        elif mode == 2: # Randomly sample a patch
            min_overlap = None

        for i in range(100):
            y, x, h, w = torchvision.transforms.RandomResizedCrop.get_params(img, self.scale_range, self.aspect_ratio_range)
            relative_x = x / img.size[0]
            relative_y = y / img.size[1]
            relative_w = w / img.size[0]
            relative_h = h / img.size[1]

            new_target = []
            for t in target:
                new_xmin = (t[1] - relative_x) / relative_w
                new_ymin = (t[2] - relative_y) / relative_h
                new_xmax = (t[3] - relative_x) / relative_w
                new_ymax = (t[4] - relative_y) / relative_h
                new_target.append([t[0], new_xmin, new_ymin, new_xmax, new_ymax])
                
            new_target = _adjust_bboxes(new_target)

            # If there is no bounding box in the cropped image, try cropping again.
            if len(new_target) == 0:
                continue

            overlap_target = random.choice(new_target)
            if min_overlap != None and (overlap_target[3] -overlap_target[1]) * (overlap_target[4] - overlap_target[2]) < min_overlap:
                continue

            img = img.crop((x, y, x + w, y + h))
            img = img.resize((self.input_size, self.input_size), self.interpolation)
            return img, new_target

        # Fall back to ODResize.
        img, target = ODResize(self.input_size)(img, target)
        img, target = ODCenterCrop(self.input_size)(img, target)
        return img, target
    
class Transform(object):
    def __call__(self, img, target=None):
        if target:
            for t in self.transforms:
                img, target = t(img, target)
            return img, target
        else:
            for t in self.transforms:
                img = t(img)
            return img

def normalize4d(tensor, mean, std):
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    tensor = tensor.clone()
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor

class InceptionTransform(Transform):
    def __init__(self, input_size):
        self.transforms = [torchvision.transforms.RandomResizedCrop(input_size),
                           torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05),
                           torchvision.transforms.RandomHorizontalFlip(),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize([0.482, 0.459, 0.408], [1,1,1])]

class IrisTransform(Transform):
    def __init__(self, input_size):
        self.transforms = [torchvision.transforms.Resize(256),
                           torchvision.transforms.TenCrop(input_size),
                           torchvision.transforms.Lambda(lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) for crop in crops])),
                           torchvision.transforms.Lambda(lambda tensor: normalize4d(tensor, [0.482, 0.459, 0.408], [1, 1, 1]))]

class CenterCropTransform(Transform):
    def __init__(self, input_size):
        self.transforms = [torchvision.transforms.Resize(input_size),
                           torchvision.transforms.CenterCrop(input_size),
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize([0.482, 0.459, 0.408], [1, 1, 1])]

class IrisODTransform(Transform):
    def __init__(self, input_size):
        self.transforms = [ODImageTransform(torchvision.transforms.functional.to_tensor),
                           ODImageTransform(lambda x : x.numpy()),
                           ODImageTransform(lambda x : x.permute((1, 2, 0))),
                           ODImageTransform(ODImResize()),
                           #ODImageTransform(torchvision.transforms.Resize(input_size)),
                           #ODCenterCrop(input_size),
                           ODImageTransform(torchvision.transforms.ToTensor()),
                           ODImageTransform(torchvision.transforms.Normalize([0.482, 0.459, 0.408], [1/255.0, 1/255.0, 1/255.0]))]

class SSDTransform(Transform):
    def __init__(self, input_size):
        self.transforms = [ODRandomExpand((0.482, 0.459, 0.408)),
                           ODRandomResizedCrop(input_size),
                           ODRandomHorizontalFlip(),
                           ODImageTransform(torchvision.transforms.ColorJitter(hue=0.05, saturation=0.05)),
                           ODImageTransform(torchvision.transforms.ToTensor()),
                           ODImageTransform(torchvision.transforms.Normalize([0.482, 0.459, 0.408], [1, 1, 1]))]
