# TODO: current design has code duplication --> need to design 
# an abstract class WrappingSampler and two sublcasses that extend it

import random
from torch.utils.data import Sampler


class SequentialWrappingSampler(Sampler):
    r"""Samples elements sequentially, in the order of dataset.
    If the required_epoch_length is bigger than the data_source length,
    the sampler will fetch from the beginning of dataset

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
        self.__start = 0 
        self.__length = max(required_epoch_length, len(self.data_source))

    def __iter__(self):
        indexes = [ind % len(self.data_source) for ind in range(self.__start, self.__stop)]
        self.__start = self.__stop % len(self.data_source)
        return iter(indexes)
    
    @property
    def __stop(self):
        return self.__start + self.__length

    def __len__(self):
        return self.__length


class RandomWrappingSampler(Sampler):
    r"""Samples elements randomly from the full dataset.
    It ensures that each batch within epoch has the same size, by setting the epoch size
    to the required_epoch_length (See SequentialWrappingSampler)
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
        self.data_source = data_source
        self.__start = 0 
        self.__length = max(required_epoch_length, len(self.data_source))

    def __iter__(self):
        indexes = [ind % len(self.data_source) for ind in range(self.__start, self.__stop)]
        random.shuffle(indexes)
        self.__start = self.__stop % len(self.data_source)
        return iter(indexes)
    
    @property
    def __stop(self):
        return self.__start + self.__length

    def __len__(self):
        return self.__length