import os
import sys
import torch
import torch.utils.data
from azure.storage.blob import BlockBlobService
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import numpy as np

# Dictionary key
IMAGE = "image"  # TODO: change to get that as parameter from prototxt
LABEL = "bbox"   # TODO: change to get that as parameter from prototxt

def _keep_max_num_bboxes(bboxes):
    """Discards boxes beyond num_bboxes"""
    num_bboxes = 30
    cur_num = bboxes.shape[0]
    diff_to_max = num_bboxes - cur_num
    if diff_to_max > 0:
        bboxes = np.lib.pad(bboxes, ((0, diff_to_max), (0, 0)),
                            "constant", constant_values=(0.0,))
    elif diff_to_max < 0:
        bboxes = bboxes[:num_bboxes, :]
    return bboxes

class AzureBlobODDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, accountName, containerName, dataset, sasToken, imageManifests, transform=None, predict_phase=False):
        self.container_name = containerName
        self.block_blob_service = BlockBlobService(account_name=accountName, sas_token=sasToken)
        self.image_manifests = imageManifests
        self.transform = transform
        self.dataset = dataset
        self.predict_phase=predict_phase
        print("Init Azureblob Dataset, IRIS imdb phase {}".format(predict_phase))
        print(len(imageManifests))
    
    def __getitem__(self, index):
        image_manifest = self.image_manifests[index]
        target = image_manifest["Regions"]
        image = None
        blobName = "{0}/{1}".format(self.dataset, image_manifest["name"])
        try:
            image = self._load_image(blobName)
        except Exception as e:
            print("Failed to load an image: {}".format(blobName))
            sys.stdout.flush()
            raise e

        if self.predict_phase:
            sample=image
            iris_target = []
            dummy_target = [True]
            w, h = image.size
            for i, t in enumerate(target):
                bbox = t["BoundingBox"]
                iris_target.append((int(t['tagIndex']), bbox[0] * w, bbox[1] * h, (bbox[0] + bbox[2]) * w, (bbox[1] + bbox[3]) * h))
            sample = self.transform(sample, dummy_target)
            return sample[0], index, h, w, iris_target
        else:
            # Convert absolute coordinates to (x1, y1, x2, y2)
            '''
            w, h = image.size
            abs_target = [None] * len(target)
            for i, t in enumerate(target):
                bbox = t["BoundingBox"]
                abs_target[i] = [bbox[0] * w, bbox[1] * h, (bbox[0] + bbox[2]) * w, (bbox[1] + bbox[3]) * h, t['tagIndex']]
            target = np.array(abs_target)
            sample = {IMAGE: image, LABEL:target}
            sample = self.transform(sample)
            return sample[IMAGE], sample[LABEL]
            '''
            sample=image
            w, h = image.size
            iris_target = []
            for i, t in enumerate(target):
                bbox = t["BoundingBox"]
                iris_target.append((int(t['tagIndex']), bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            sample, iris_target = self.transform(sample, iris_target)
            target = []
            for t in iris_target:
                target.append([t[1] * w, t[2] * h, t[3] * w, t[4] * h, t[0]])
            target = np.array(target)
            target = _keep_max_num_bboxes(target).flatten()
            return sample, target

    def __len__(self):
        return len(self.image_manifests)

    def _load_image(self, blobName):
        data = self.block_blob_service.get_blob_to_bytes(self.container_name, blobName).content
        image = Image.open(BytesIO(data))
        return image.convert('RGB')
