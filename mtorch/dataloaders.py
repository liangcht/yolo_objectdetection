import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler

from mmod.dist_utils import env_world_size, env_rank
from mtorch.imdbdata import ImdbData
from mtorch.tbox_utils import Labeler
from mtorch.augmentation import DefaultDarknetAugmentation, TestAugmentation
from mtorch.distributed_samplers import DistributedSequentialWrappingSampler, DistributedRandomWrappingSampler

__all__ = ['yolo_train_data_loader', 'yolo_test_data_loader']

WRAPPING = True
SEQUENTIAL_SAMPLER = True
RANDOM_SAMPLER = not SEQUENTIAL_SAMPLER


def _list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader,
    if you want to have a list of items as an output, as opposed to tensors
    """
    items = list(zip(*batch))
    return items


def yolo_train_data_loader(datafile, batch_size=16, num_workers=2, distributed=True):
    """prepares data loader for training
    :param datafile: str, path to file with data
    :param batch_size: int, batch size per GPU
    :param num_workers: int, number of workers per GPU
    :param distributed: bool, if distributed training is used
    :return: data loader
    """
    augmenter = DefaultDarknetAugmentation()
    augmented_dataset = ImdbData(path=datafile,
                                 transform=augmenter(),
                                 labeler=Labeler())

    if WRAPPING:
        total_batch_size = batch_size * env_world_size()
        full_epoch_dataset_length = int(np.ceil(float(len(augmented_dataset)) / float(total_batch_size))) \
                                    * \
                                    total_batch_size
        if SEQUENTIAL_SAMPLER:
            sampler = DistributedSequentialWrappingSampler(
                augmented_dataset,
                full_epoch_dataset_length,
                num_replicas=env_world_size() if distributed else 1,
                rank=env_rank() if distributed else 0
            )
        else:  # use RANDOM_SAMPLER
            sampler = DistributedRandomWrappingSampler(
                augmented_dataset,
                full_epoch_dataset_length,
                num_replicas=env_world_size() if distributed else 1,
                rank=env_rank() if distributed else 0
            )
    else:  # use native PyTorch Distributed Sampler
        sampler = DistributedSampler(
            augmented_dataset,
            num_replicas=env_world_size() if distributed else 1,
            rank=env_rank() if distributed else 0
        )

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True)


def yolo_test_data_loader(datafile, batch_size=1, num_workers=2):
    """prepares test data loader
    :param datafile: str, path to file with data
    :param batch_size: int, batch size per GPU
    :param num_workers: int, number of workers
    :return: data loader
    """
    test_augmenter = TestAugmentation()
    augmented_dataset = ImdbData(path=datafile,
                                 transform=test_augmenter(),
                                 predict_phase=True)
    sampler = SequentialSampler(augmented_dataset)

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True,
                      collate_fn=_list_collate)
