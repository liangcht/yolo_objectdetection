import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate

from mmod.dist_utils import env_world_size, env_rank
from mtorch.imdbdata import ImdbData
from mtorch.imdbregions import ImdbRegions
from mtorch.tbox_utils import Labeler, ClassLabeler, RegionCropper
from mtorch.augmentation import DefaultDarknetAugmentation, TestAugmentation, ClassifierTrainAugmentation, ClassifierTestAugmentation
from mtorch.distributed_samplers import DistributedSequentialWrappingSampler, DistributedRandomWrappingSampler, DistributedBalancedSampler
from mtorch.region_conditions import HasConfAbove, HasHeightAbove, HasWidthAbove

__all__ = ['yolo_train_data_loader', 'yolo_test_data_loader']


def _list_collate(batch):
    """ Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader,
    if you want to have a list of items as an output, as opposed to tensors
    """
    items = list(zip(*batch))
    return items


def create_imdb_dataset(path, cmapfile, transform, labeler=None, **kwargs):
    if '$' in path:
        from mtorch.imdbtsvdata import ImdbTSVData
        return ImdbTSVData(path, cmapfile, transform, labeler, **kwargs)
    else:
        return ImdbData(path, cmapfile, transform, labeler, **kwargs)


def yolo_train_data_loader(args):
    """prepares data loader for training
    :param args: dict, of input arguments
    :return: data loader
    """
    datafile = args["train"]
    cmapfile = args["labelmap"]
    try:
        batch_size = args["batch_size"]
    except KeyError:
        batch_size = 16
    try:
        num_workers = args["workers"]
    except KeyError:
        num_workers = 2
    try:
        distributed = args["distributed"]
    except KeyError:
        distributed = False
    try:
        use_wrap_sampler = args["wrap"]
    except KeyError:
        use_wrap_sampler = True
    try:
        use_random_sampler = args["random"]
    except KeyError:
        use_random_sampler = False
    try:
        min_iters_per_epoch = args["min_iters_in_epoch"]
    except KeyError:
        min_iters_per_epoch = 4 * 70

    augmenter = DefaultDarknetAugmentation()
    augmented_dataset = create_imdb_dataset(datafile,
            cmapfile, augmenter(), Labeler())

    if use_wrap_sampler:
        total_batch_size = batch_size * env_world_size()
        num_iters_per_epoch = max(min_iters_per_epoch,
                                int(np.ceil(float(len(augmented_dataset)) / float(total_batch_size)))) 
        full_epoch_dataset_length = num_iters_per_epoch * total_batch_size
        if not use_random_sampler:  # use SEQUENTIAL SAMPLER
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


def yolo_test_data_loader(datafile, cmapfile=None, batch_size=1, num_workers=2):
    """prepares test data loader
    :param datafile: str, path to file with data
    :param batch_size: int, batch size per GPU
    :param num_workers: int, number of workers
    :return: data loader
    """
    test_augmenter = TestAugmentation()
    augmented_dataset = create_imdb_dataset(path=datafile,
                                 cmapfile=cmapfile,
                                 transform=test_augmenter(),
                                 predict_phase=True)

    sampler = SequentialSampler(augmented_dataset)

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True,
                      collate_fn=_list_collate)

