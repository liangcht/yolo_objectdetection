import cv2
import numpy as np
from mmod.im_utils import im_rescale

import cv2
class ODImResize(object):

    def __init__(self, target_size=416, maintain_ratio=True):
        self.target_size = target_size
        self.maintain_ratio = maintain_ratio

    def __call__(self, im):
        self._set_network_input_size(*im.shape[0:2])   
        im_resized = im_rescale(im.astype(np.float32, copy=True), 
                                max(self.network_input_width, self.network_input_height))
        new_h, new_w = im_resized.shape[0:2]
        left = (self.network_input_width - new_w) / 2
        right = self.network_input_width - new_w - left
        top = (self.network_input_height - new_h) / 2
        bottom = self.network_input_height - new_h - top
        im_squared = cv2.copyMakeBorder(im_resized, top=top, bottom=bottom, left=left, right=right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
 
        return im_squared
    
    def _set_network_input_size(self, h, w):
        if self.maintain_ratio:
            alpha = self.target_size / np.sqrt(h * w)
            height2 = int(np.round(alpha * h))
            width2 = int(np.round(alpha * w))
            if h > w:
                self.network_input_height = (height2 + 31) / 32 * 32
                self.network_input_width = ((self.network_input_height * w + h - 1) / h
                                       + 31) / 32 * 32
            else:
                self.network_input_width = (width2 + 31) / 32 * 32
                self.network_input_height = ((self.network_input_width * h + w - 1) / w +
                                        31) / 32 * 32
        else:
            self.network_input_width = self.target_size
            self.network_input_height = self.target_size

