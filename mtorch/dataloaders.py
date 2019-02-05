import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mmod.dist_utils import env_world_size, env_rank
from mtorch.imdbdata import ImdbData
from mtorch.tbox_utils import Labeler
from mtorch.augmentation import DefaultDarknetAugmentation
from mtorch.samplers import SequentialWrappingSampler, RandomWrappingSampler
from torch.utils.data import SequentialSampler


def yolo_train_data_loader(datafile,batch_size=16, num_workers=2, distributed=True):
    augmenter = DefaultDarknetAugmentation()
    augmented_dataset = ImdbData(path=datafile,
                                 transform=augmenter(),
                                 labeler=Labeler())

    # sampler = SequentialWrappingSampler(
    #     augmented_dataset,
    #     int(np.ceil(float(len(augmented_dataset)) / float(batch_size)) * batch_size)
    # )
    sampler = SequentialSampler(
         augmented_dataset)
    if distributed:
        sampler = DistributedSampler(augmented_dataset, num_replicas=env_world_size(), rank=env_rank() if distributed else None)

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True)


