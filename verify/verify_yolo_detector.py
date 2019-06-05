import os
import time

import cv2
import numpy as np
import pdb
from mtorch.yolo_detector import YoloDetector

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# path_model = '/home/t-apsi/apar/test/files/model_epoch_36.pt'
# path_labelmap = '/home/t-apsi/apar/test/files/labelmap0.txt'
path_model = '/home/tobai/ODExperiments/objectdetection/output/_epoch_10.pt'
print(path_model)
path_labelmap = '/home/tobai/ODExperiments/dataset/benchmark_dataset/animal661/Animal-661_labels.txt'

detector = YoloDetector(path_model, path_labelmap)

# path_images  = '/home/t-apsi/apar/test/images'
# for f, filename in enumerate(os.listdir(path_images)):
#     print(filename)
#     image = cv2.imread(os.path.join(path_images, filename))
#     res = detector.detect(image)
#     print(res)

    # Save the images with bounding boxes
    # for i, bbox in enumerate(res):
    #     bbox['rect'] = [int(x) for x in bbox['rect']]
    #     x1, y1, x2, y2 = bbox['rect']
    #     temp = image.copy()
    #     temp[y1-1:y1+1, x1:x2, :] = 0
    #     temp[y2-1:y2+1, x1:x2, :] = 0
    #     temp[y1:y2, x1-1:x1+1, :] = 0
    #     temp[y1:y2, x2-1:x2+1, :] = 0

    #     output_name = 'img' + '_' + bbox['class'] + '_' + str(int(bbox['conf']*100))
    #     cv2.imwrite(os.path.join('test', output_name +'.jpg'), temp)

path_image = "/home/tobai/ODExperiments/dataset/benchmark_dataset/animal661/train_images/170.jpg"
image = cv2.imread(path_image)
res = detector.detect(image)
res = sorted(res, key = lambda det: det['conf'])
print(res[-5:])

# with open("/home/tobai/ODExperiments/dataset/benchmark_dataset/animal661/train_label/170.txt") as f:
#     gt = f.readlines()
# temp = image.copy()
# print(gt)
# for line in gt:
#     bbox = line.strip().split()
#     x1, y1, x2, y2 = int(round(float(bbox[-4]))),int(round(float(bbox[-3]))),int(round(float(bbox[-2]))), int(round(float(bbox[-1])))
    
#     temp[y1-1:y1+1, x1:x2, :] = 0
#     temp[y2-1:y2+1, x1:x2, :] = 0
#     temp[y1:y2, x1-1:x1+1, :] = 0
#     temp[y1:y2, x2-1:x2+1, :] = 0

# output_name = 'img_gt'
# out_pth = os.path.join('test', output_name +'.jpg')
# print(out_pth)
# cv2.imwrite(out_pth, temp)

# Save the images with bounding boxes
for i, bbox in enumerate(res):
    if (bbox['conf']> 0.2):
        bbox['rect'] = [int(x) for x in bbox['rect']]
        x1, y1, x2, y2 = bbox['rect']
        temp = image.copy()
        temp[y1-1:y1+1, x1:x2, :] = 0
        temp[y2-1:y2+1, x1:x2, :] = 0
        temp[y1:y2, x1-1:x1+1, :] = 0
        temp[y1:y2, x2-1:x2+1, :] = 0

        output_name = 'img' + '_' + bbox['class'] + '_' + str(int(bbox['conf']*100))
        out_pth = os.path.join('test', output_name +'.jpg')
        print(out_pth)
        cv2.imwrite(out_pth, temp)
