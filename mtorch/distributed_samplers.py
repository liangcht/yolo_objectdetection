import math
import torch
from torch.utils.data.distributed import DistributedSampler

__all__ = ['DistributedSequentialWrappingSampler', 'DistributedRandomWrappingSampler']


class DistributedWrappingSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    To ensure compatibility with caffe:
    It wraps around the dataset to ensure equal size batches within an epoch
    and per GPU/nodes/process (see also WrappingSampler if distribution is not required).

    Parameters:
        dataset: Dataset used for sampling.
        required_epoch_length: int, the required length of dataset to ensure
        equally sized batches within
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, required_epoch_length, num_replicas=None, rank=None):
        """This is and Abstract class - do NOT instantiate, use concrete classes:
        DistributedSequentialWrappingDataset or DistributedRandomSequentialDataset
        :param dataset:  the dataset to sample
        :param required_epoch_length: required number of samples in epoch
        :param num_replicas: number of GPUs/nodes/processes
        :param rank: rank of this particular GPU/node/process
        """
        super(DistributedWrappingSampler, self).__init__(dataset, num_replicas=num_replicas, rank=rank)
        self._start = 0
        self._length = max(required_epoch_length, len(self.dataset))
        self.num_samples = int(math.ceil(self._length * 1.0 / self.num_replicas))
        self._length = self.num_samples * self.num_replicas

    def __iter__(self):
        """iterator
        :return: indexes per replica
        """
        indexes = self._choose_indexes()
        self._set_start_of_next_epoch()
        indexes = self._subsample(indexes)
        return iter(indexes)

    def _choose_indexes(self):
        """helper to choose indexes - differently implemented in each concrete class"""
        raise NotImplementedError(
            "You need to create an instance of either DistributedSequentialWrappingSampler"
            " or DistributedRandomWrappingSampler "
        )

    def _subsample(self, indexes):
        """helper to subsample indexes"""
        return indexes[self.rank: self._length: self.num_replicas]

    @property
    def _stop(self):
        """helper to determine the stop location within the dataset"""
        return self._start + self._length

    def _set_start_of_next_epoch(self):
        """helper to set start of next epoch (right after the end of this epoch"""
        self._start = self._stop % len(self.dataset)


class DistributedSequentialWrappingSampler(DistributedWrappingSampler):
    """Sequential Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    To ensure compatibility with caffe:
    It wraps around the dataset to ensure equal size batches within an epoch
    and per GPU/nodes/process (see also WrappingSampler if distribution is not required).
    """

    def __init__(self, data_source, required_epoch_length, num_replicas=None, rank=None):
        """ Constructs the object of DistributedSequentialWrappingSampler
        :param data_source:  the data_source to sample
        :param required_epoch_length: required number of samples in epoch
        :param num_replicas: number of GPUs/nodes/processes
        :param rank: rank of this particular GPU/node/process
        """
        super(DistributedSequentialWrappingSampler, self).__init__(data_source, required_epoch_length,
                                                                   num_replicas=num_replicas, rank=rank)

    def _choose_indexes(self):
        """helper to subsample indexes"""
        return [ind % len(self.dataset) for ind in range(self._start, self._stop)]


class DistributedRandomWrappingSampler(DistributedWrappingSampler):
    """Random Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    To ensure compatibility with caffe:
    It wraps around the dataset to ensure equal size batches within an epoch
    and per GPU/nodes/process (see also WrappingSampler if distribution is not required).
    """

    def __init__(self, data_source, required_epoch_length, num_replicas=None, rank=None):
        """ Constructs the object of DistributedRandomWrappingSampler
        :param data_source:  the data_source to sample
        :param required_epoch_length: required number of samples in epoch
        :param num_replicas: number of GPUs/nodes/processes
        :param rank: rank of this particular GPU/node/process
        """
        super(DistributedRandomWrappingSampler, self).__init__(data_source, required_epoch_length,
                                                               num_replicas=num_replicas, rank=rank)

    def _choose_indexes(self):
        """helper to subsample indexes"""
        g = torch.Generator()
        g.manual_seed(self.epoch)  # deterministically shuffle 
        indexes = [ind % len(self.dataset) for ind in range(self._start, self._stop)]
        random_inds = torch.randperm(len(indexes), generator=g).tolist()
        return [indexes[random_ind] for random_ind in random_inds]
