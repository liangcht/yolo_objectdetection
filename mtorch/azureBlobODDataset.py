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
            image, targets = self.transform(image, targets)
            w, h = image.size
            return image, index, h, w, targets
        else:
            # Convert absolute coordinates to (x1, y1, x2, y2)
            ltwh_target = [None] * len(targets)
            for i, t in enumerate(targets):
                bbox = t["BoundingBox"]

                ltwh_target[i] = [t['tagIndex'], bbox[0], bbox[1], bbox[2], bbox[3]]
            #targets = np.array(abs_target)
            image, targets = self.transform(image, ltwh_target)
            print(image)
            print(targets)
            targets = np.asarray(targets)
            targets = _keep_max_num_bboxes(targets).flatten()
            sample = {IMAGE: image, LABEL:targets}
            
            return sample[IMAGE], sample[LABEL]

    def __len__(self):
        return len(self.image_manifests)

    def _load_image(self, blobName):
        data = self.block_blob_service.get_blob_to_bytes(self.container_name, blobName).content
        image = Image.open(BytesIO(data))
        return image.convert('RGB')
