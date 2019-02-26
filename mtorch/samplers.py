import random
from torch.utils.data import Sampler

__all__ = ['SequentialWrappingSampler', 'RandomWrappingSampler']


class WrappingSampler(Sampler):
    """Abstract Class Wrapping Sampler
    If the required_epoch_length is bigger than the data_source length,
    the sampler will fetch from the beginning of dataset, maintaining cirularity 

    Arguments:
        data_source (Dataset): dataset to sample from
        required_epoch_length (int): number of samples in
    """

    def __init__(self, data_source, required_epoch_length):
        """ Constructs the object of SequentialWrappingSampler
        :param data_source: data set to sample from
        :param required_epoch_length: the length of the epoch
        (if less then length of dataset, the dataset length will be used)
        """
        self.data_source = data_source
        self._start = 0
        self._length = max(required_epoch_length, len(self.data_source))

    def __iter__(self):
        indexes = self._choose_indexes()
        self._set_start_of_next_epoch()
        return iter(indexes)

    @property
    def _stop(self):
        return self._start + self._length

    def __len__(self):
        return self._length

    def _choose_indexes(self):
        raise NotImplementedError(
            "You need to create an instance of either SequentialWrappingSampler or RandomWrappingSampler "
        )

    def _set_start_of_next_epoch(self):
        self._start = self._stop % len(self.data_source)


class SequentialWrappingSampler(WrappingSampler):
    """Samples elements sequentially, in the order of dataset.
    If the required_epoch_length is bigger than the data_source length,
    the sampler will fetch from the beginning of dataset (Extends WrappingSampler)

    Arguments:
        data_source (Dataset): dataset to sample from
        required_epoch_length (int): number of samples in
    """

    def __init__(self, data_source, required_epoch_length):
        """ Constructs the object of SequentialWrappingSampler
        :param data_source: data set to sample from
        :param required_epoch_length: the length of the epoch
        (if less then length of dataset, the dataset length will be used)
        """
        super(SequentialWrappingSampler, self).__init__(data_source, required_epoch_length)

    def _choose_indexes(self):
        return [ind % len(self.data_source) for ind in range(self._start, self._stop)]


class RandomWrappingSampler(WrappingSampler):
    """Samples elements randomly from the full dataset.
    It ensures that each batch within epoch has the same size, by setting the epoch size
    to the required_epoch_length (Extends WrappingSampler)
    Arguments:
        data_source (Dataset): dataset to sample from
        required_epoch_length (int): number of samples in
    """

    def __init__(self, data_source, required_epoch_length):
        """ Constructs the object of RandomWrappingSampler
        :param data_source: data set to sample from
        :param required_epoch_length: the length of the epoch
        (if less then length of dataset, the dataset length will be used)
        """
        super(RandomWrappingSampler, self).__init__(data_source, required_epoch_length)

    def _choose_indexes(self):
        indexes = [ind % len(self.data_source) for ind in range(self._start, self._stop)]
        random.shuffle(indexes)
        return indexes
