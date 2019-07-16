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

class IRISAzureBlobODDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, accountName, containerName, dataset, sasToken, imageManifests, transform=None):
        self.container_name = containerName
        self.block_blob_service = BlockBlobService(account_name=accountName, sas_token=sasToken)
        self.image_manifests = imageManifests
        self.transform = transform
        self.dataset = dataset
        print(len(imageManifests))
    
    def __getitem__(self, index):
        image_manifest = self.image_manifests[index]
        target = image_manifest["Regions"]
        image = None
        blobName = "{0}/{1}".format(self.dataset, image_manifest["Id"].replace('-', ''))
        try:
            image = self._load_image(blobName)
        except Exception as e:
            print("Failed to load an image: {}".format(blobName))
            sys.stdout.flush()
            raise e

        sample=image
        iris_target = []
        for i, t in enumerate(target):
            bbox = t["BoundingBox"]
            iris_target.append((int(t['TagIndex']), bbox['Left'], bbox['Top'], (bbox['Left'] + bbox['Width']), (bbox['Top'] + bbox['Height'])))
        sample, target = self.transform(sample, iris_target)
        return sample, target

    def __len__(self):
        return len(self.image_manifests)

    def _load_image(self, blobName):
        data = self.block_blob_service.get_blob_to_bytes(self.container_name, blobName).content
        image = Image.open(BytesIO(data))
        return image.convert('RGB')
