import cv2
import numpy as np
from mmod.im_utils import im_rescale


class ODImResize(object):
    """Resizing transform of an image during inference
    """

    def __init__(self, target_size=(416, 416), maintain_ratio=True):
        """Constructor of Resizing transform for inference
        :param target_size: ideal size of an image (height, width) or scalar if height = width
        :param maintain_ratio: if to maintain aspect ratio while testing
        """
        if not isinstance(target_size, tuple):
            target_size = tuple(target_size, target_size)
        self.target_size = target_size
        self.maintain_ratio = maintain_ratio

    def __call__(self, im):
        self._set_network_input_size(*im.shape[0:2])   
        im_resized = im_rescale(im.astype(np.float32, copy=True), 
                                max(self.network_input_width, self.network_input_height))
        new_h, new_w = im_resized.shape[0:2]
        left = int(np.round((self.network_input_width - new_w) / 2))
        right = int(np.round(self.network_input_width - new_w - left))
        top = int(np.round((self.network_input_height - new_h) / 2))
        bottom = int(np.round((self.network_input_height - new_h - top)))
        im_squared = cv2.copyMakeBorder(im_resized, top=top, bottom=bottom, left=left, right=right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        print(im_squared.shape[0:2])
        return im_squared

    def _set_network_input_size(self, h, w):
        if self.maintain_ratio:
            alpha = np.sqrt(self.target_size[0] * self.target_size[1]) / np.sqrt(h * w)
            height2 = int(np.round(alpha * h))
            width2 = int(np.round(alpha * w))
            if h > w:
                self.network_input_height = (height2 + 31) // 32 * 32
                self.network_input_width = ((self.network_input_height * w + h - 1) // h +
                                            31) // 32 * 32
            else:
                self.network_input_width = (width2 + 31) // 32 * 32
                self.network_input_height = ((self.network_input_width * h + w - 1) // w +
                                            31) // 32 * 32
        else:
            self.network_input_height = self.target_size[0]
            self.network_input_width = self.target_size[1]

    '''
    def _set_network_input_size(self, h, w):
        if h > w:
            self.network_input_height = ((416.0 * h / w) + 31) // 32 * 32
            self.network_input_width = 416
        else:
            self.network_input_width =  ((416.0 * w / h) + 31) // 32 * 32
            self.network_input_height = 416
    '''