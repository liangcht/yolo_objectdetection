import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate

from mmod.dist_utils import env_world_size, env_rank
from mtorch.imdbdata import ImdbData
from mtorch.imdbregions import ImdbRegions
from mtorch.tbox_utils import Labeler, ClassLabeler, RegionCropper
from mtorch.augmentation import ClassifierTrainAugmentation, ClassifierTestAugmentation
from mtorch.distributed_samplers import DistributedSequentialWrappingSampler, DistributedRandomWrappingSampler
from mtorch.region_conditions import HasConfAbove, HasHeightAbove, HasWidthAbove

__all__ = ['region_classifier_data_loader', 'region_classifier_test_data_loader']

WRAPPING = True
SEQUENTIAL_SAMPLER = False
RANDOM_SAMPLER = not SEQUENTIAL_SAMPLER
MIN_ITERS_PER_EPOCH = 10  # TODO: read from config (TBD)


def _list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader,
    if you want to have a list of items as an output, as opposed to tensors
    """
    items = list(zip(*batch))
    return items


def _classifier_test_collate(batch):
    imgs = default_collate([item[0] for item in batch])
    keys = [item[1] for item in batch]
    img_keys = [item[2] for item in batch]
    bbox_rects = [item[3] for item in batch]
    return [imgs, keys, img_keys, bbox_rects]


def _get_train_sampler(augmented_dataset, batch_size, distributed=True):
    if WRAPPING:
        total_batch_size = batch_size * (env_world_size() if distributed else 1)
        num_iters_per_epoch = max(MIN_ITERS_PER_EPOCH,
                                  int(np.ceil(float(len(augmented_dataset)) / float(total_batch_size))))
        full_epoch_dataset_length = num_iters_per_epoch * total_batch_size
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
        print("Using distributed")

        sampler = DistributedSampler(
            augmented_dataset,
            num_replicas=env_world_size() if distributed else 1,
            rank=env_rank() if distributed else 0
        )
    return sampler


def region_classifier_data_loader(datafile, pos_conf=0.1,
                                  cmapfile=None, batch_size=16, num_workers=2, distributed=True):
    """prepares data loader for training
    :param datafile: str, path to file with data
    :param pos_conf: float, minimal confidence to be considered to be a positive example
    :param cmapfile: str, path to the file which contains class information
    :param batch_size: int, batch size per GPU
    :param num_workers: int, number of workers per GPU
    :param distributed: bool, if distributed training is used
    :return: data loader
    """
    augmenter = ClassifierTrainAugmentation()
    region_cropper = RegionCropper([HasWidthAbove(), HasHeightAbove()])
    augmented_dataset = ImdbRegions(path=datafile,
                                    region_cropper=region_cropper,
                                    cmapfile=cmapfile,
                                    transform=augmenter(),
                                    labeler=ClassLabeler(HasConfAbove(pos_conf)))

    sampler = _get_train_sampler(augmented_dataset, batch_size, distributed)

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True)


def region_classifier_test_data_loader(datafile, cmapfile=None, batch_size=1, num_workers=2):
    """prepares test data loader
    :param datafile: str, path to file with data
    :param cmapfile: str, path to the file which contains class information
    :param batch_size: int, batch size per GPU
    :param num_workers: int, number of workers
    :return: data loader
    """
    test_augmenter = ClassifierTestAugmentation()
    region_cropper = RegionCropper([HasWidthAbove(0), HasHeightAbove(0)])

    augmented_dataset = ImdbRegions(path=datafile,
                                    region_cropper=region_cropper,
                                    cmapfile=cmapfile,
                                    transform=test_augmenter(),
                                    predict_phase=True)
    sampler = SequentialSampler(augmented_dataset)

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True,
                      collate_fn=_classifier_test_collate)
