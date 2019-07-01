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
    
    def __getitem__(self, index):
        image_manifest = self.image_manifests[index]
        targets = image_manifest["Regions"]
        image = None
        blobName = "{0}/{1}".format(self.dataset, image_manifest["name"])
        try:
            image = self._load_image(blobName)
        except Exception as e:
            print("Failed to load an image: {}".format(blobName))
            sys.stdout.flush()
            raise e

        if self.predict_phase:
            w, h = image.size
            iris_target = [None] * len(targets)
            for i, t in enumerate(targets):
                bbox = t["BoundingBox"]

                iris_target[i] = [t['tagIndex'], bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            image, targets = self.transform(image, iris_target)
            return image, index, h, w, targets
        else:
            # Convert absolute coordinates to (x1, y1, x2, y2)
            iris_target = [None] * len(targets)
            for i, t in enumerate(targets):
                bbox = t["BoundingBox"]

                iris_target[i] = [t['tagIndex'], bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            image, targets = self.transform(image, iris_target)
            np_target = np.zeros(shape=(len(targets), 5), dtype=float)
            for i, t in enumerate(targets):
                np_target[i] = np.asarray(t)
            targets = _keep_max_num_bboxes(np_target).flatten()
            sample = {IMAGE: image, LABEL:targets}
            
            return sample[IMAGE], sample[LABEL]

    def __len__(self):
        return len(self.image_manifests)

    def _load_image(self, blobName):
        data = self.block_blob_service.get_blob_to_bytes(self.container_name, blobName).content
        image = Image.open(BytesIO(data))
        return image.convert('RGB')
