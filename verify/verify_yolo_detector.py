import os
import time

import cv2
import numpy as np

from mtorch.yolo_detector import YoloDetector

os.environ['CUDA_VISIBLE_DEVICES'] = "4"


path_model = '/home/t-apsi/apar/test/files/model_epoch_36.pt'
path_labelmap = '/home/t-apsi/apar/test/files/labelmap0.txt'

path_images  = '/home/t-apsi/apar/test/images'

detector = YoloDetector(path_model, path_labelmap, thresh=0, obj_thresh=0)
for f, filename in enumerate(os.listdir(path_images)):
    print(filename)
    image = cv2.imread(os.path.join(path_images, filename))
    res = detector.detect(image)
    print(res)

    # Save the images with bounding boxes
    # for i, bbox in enumerate(res):
    #     bbox['rect'] = [int(x) for x in bbox['rect']]
    #     x1, y1, x2, y2 = bbox['rect']
    #     temp = image.copy()
    #     temp[y1-1:y1+1, x1:x2, :] = 0
    #     temp[y2-1:y2+1, x1:x2, :] = 0
    #     temp[y1:y2, x1-1:x1+1, :] = 0
    #     temp[y1:y2, x2-1:x2+1, :] = 0

    #     output_name = filename + '_' + bbox['class'] + '_' + str(int(bbox['conf']*100))
    #     cv2.imwrite(os.path.join('test', 'saved', output_name +'.jpg'), temp)
