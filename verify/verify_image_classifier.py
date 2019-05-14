import os
import cv2
import time
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

from mtorch.image_classifier import ImageClassifier

path_model = '/home/t-apsi/apar/test/files_classifier/with_background_v3_resnet_lr_0.01_decay_0.01_resize_256-0101.pth.tar'
path_labelmap = '/home/t-apsi/apar/test/files_classifier/labelmap.txt'

path_images  = '/home/t-apsi/apar/test/images'
rect = [47.98457087753134,59.537275064267355,160.1542912246866,140.82262210796915]
classifier = ImageClassifier(path_model, path_labelmap)
for f, filename in enumerate(os.listdir(path_images)):
    print(filename)
    image = cv2.imread(os.path.join(path_images, filename))
    res = classifier.predict(image, rect=rect, topk=1)
    print(res)
