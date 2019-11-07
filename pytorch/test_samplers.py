import torch
from torch._six import int_classes as _int_classes


[docs]class Sampler(object):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError


    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


[docs]class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)



[docs]class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples



[docs]class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)



[docs]class WeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples



[docs]class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

######################################

class WeightedSubsetRandomSampler(Sampler):
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, indices, weights, num_samples=0):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool):
            raise ValueError("num_samples should be a non-negative integeral "
                             "value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [ weights[i] for i in self.indices ]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = True

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples

class MyDataset(Dataset):
        ...
     # return data idx here, so that the corresponding weight can be updated based on the training loss.
    def __getitem__(self, idx):
        ...
        return image, label, idx

sample_weights = np.ones(len(train_labels))
for epoch in range(0, total_epochs):
        # tr_indices indexes a subset of train_labels
        train_sampler = WeightedSubsetRandomSampler(tr_indices, sample_weights)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

        for batch_i, (data, batch_truth, data_indices) in enumerate(train_loader):
              ...
              batch_loss = criterion_individual(scores, batch_truth)
              # update sample weights to be the loss, so that harder samples have larger chances to be drawn in the next epoch
              sample_weights[data_indices] = batch_loss.detach().cpu().numpy()


#####################################
# 1. DATASET
class DummyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return torch.ones(5)*index

    def __len__(self):
        return 100

# 2. SAMPLER
class DynamicSampler(object):
    def __init__(self):
        self.next_batch = [0]

    def next_sample(self, indList):
        self.next_batch = indList

    def __iter__(self):
        return iter(self.next_batch)

    def __len__(self):
        return 100

# EXAMPLE
if __name__ == '__main__':
    dataset = DummyDataset()
    sampler = DynamicSampler()
    loader = DataLoader(dataset, sampler=sampler)

    best_batch = [5, 6, 7, 10, 11]
    for i in range(n_iters):
        # Set the next batch indices as the best batch
        loader.sampler.next_sample(best_batch)

        # Get Batch
        loader.batch_sampler.batch_size = len(best_batch)
        batch = iter(loader).next()

        # Update weights
        model.partial_fit(batch)

        # Get the next best batch
        best_batch = get_nextBestBatch(model, dataset)

#####################################################


#############################
# color-coded tensor print
#############################

import math
import shutil


# print string with specified color
def printk(color, s):
    if color == "red" :
        colorCode = "\033[31m"
    elif color == "green" :
        colorCode = "\033[32m"
    elif color == "yellow" :
        colorCode = "\033[1;33m"
    elif color == "orange" :
        colorCode = "\033[33m"
    elif color == "blue" :
        colorCode = "\033[34m"
    elif color == "purple" :
        colorCode = "\033[35m"
    elif color == "cyan" :
        colorCode = "\033[36m"
    elif color == "white" :
        colorCode = "\033[37m"
    elif color == "grayback" :
        colorCode = "\033[40m"
    elif color == "bold" :
        colorCode = "\033[97m"
    else:
        colorCode = "\033[0m"
    print(colorCode + s + "\033[0m")

##  colorCodes = {black, VIBGYOR, White}
def getCode(value, max, s):
    if value == 0:
        value = 0.1
    if s == None or s == '':
        s = '█'
    colorCodes = ["\033[30m", "\033[1;30m", "\033[35m", "\033[1;35m", "\033[34m", "\033[1;34m", "\033[36m", "\033[32m", "\033[1;32m", "\033[1;33m", "\033[33m",  "\033[1;31m", "\033[31m", "\033[37m", "\033[1;37m"]
    index = int((len(colorCodes)-1) * (value/max))
    return colorCodes[index] + s + "\033[0m"

def getTerminalWidth():
    default_width = 80
    default_height = 20
    size_tuple = shutil.get_terminal_size((default_width, default_height))  # pass fallback
    return size_tuple.columns

# creates numBins of equal length
# (except last bin which contains remaining items)
# returns max val in each bucket
def bucket(data, numBins):
    dataLength = len(data)
    if dataLength <= numBins:
        return data
    windowLength = dataLength//numBins
    limit = numBins * windowLength
    output = torch.Tensor(numBins)
    for i in range(0, numBins):
        start = i*windowLength
        stop = (i+1)*windowLength
        if (stop == limit):
            stop = dataLength
        output[i] = torch.max(data[start:stop])
    return output

def colorCodedPrint(data, title):
    title = "" if title == None else str(title)+":"
    barLength = getTerminalWidth() - 32 - len(title)
    normalizedData = torch.ceil((1.0*data*barLength)/torch.max(data))
    # normalizedData = (data*barLength)/torch.max(data)
    bucketData = bucket(normalizedData, barLength)
    maxValue = torch.max(bucketData)
    code = ' '
    for i in range(1, len(bucketData)):
        code = code + getCode(bucketData[i], maxValue, '█')
    print(title + code)
