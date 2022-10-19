# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import pickle
import gzip
import numpy as np
import os
from mlp import DEFAULT_SEED


class DataProvider(object):
    """Generic data provider."""

    def __init__(self, inputs, targets, batch_size, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new data provider object.

        Args:
            inputs (ndarray): Array of data input features of shape
                (num_data, input_dim).
            targets (ndarray): Array of data output targets of shape
                (num_data, output_dim) or (num_data,) if output_dim == 1.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        assert max_num_batches != 0 and not max_num_batches < -1, (
            'max_num_batches should be -1 or > 0')
        self.max_num_batches = max_num_batches
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // batch_size
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)
        self.shuffle_order = shuffle_order
        if rng is None:
            rng = np.random.RandomState(DEFAULT_SEED)
        self.rng = rng
        self.reset()

    def __iter__(self):
        """Implements Python iterator interface.

        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """
        return self

    def reset(self):
        """Resets the provider to the initial state to use in a new epoch."""
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def shuffle(self):
        """Randomly shuffles order of data."""
        new_order = self.rng.permutation(self.inputs.shape[0])
        self.inputs = self.inputs[new_order]
        self.targets = self.targets[new_order]

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so reset
            # the dataset for another pass and indicate iteration is at end
            self.reset()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch


class MNISTDataProvider(DataProvider):
    """Data provider for MNIST handwritten digit images."""

    def __init__(self, which_set='train', batch_size=100, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'eval'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        # check a valid which_set was provided
        assert which_set in ['train', 'valid', 'eval'], (
            'Expected which_set to be either train, valid or eval. '
            'Got {0}'.format(which_set)
        )
        self.which_set = which_set
        self.num_classes = 10
        # construct path to data using os.path.join to ensure the correct path
        # separator for the current platform / OS is used
        # MLP_DATA_DIR environment variable should point to the data directory
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'mnist-{0}.npz'.format(which_set))
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets = loaded['inputs'], loaded['targets']
        inputs = inputs.astype(np.float32)
        # pass the loaded data to the parent class __init__
        super(MNISTDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MNISTDataProvider, self).next()
        return inputs_batch, self.to_one_of_k(targets_batch)
    
    def __next__(self):
        return self.next()

    def to_one_of_k(self, int_targets):
        """Converts integer coded class target to 1 of K coded targets.

        Args:
            int_targets (ndarray): Array of integer coded class targets (i.e.
                where an integer from 0 to `num_classes` - 1 is used to
                indicate which is the correct class). This should be of shape
                (num_data,).

        Returns:
            Array of 1 of K coded targets i.e. an array of shape
            (num_data, num_classes) where for each row all elements are equal
            to zero except for the column corresponding to the correct class
            which is equal to one.
        """
        result = [self.__convert_to_k(target) for target in int_targets]
        return np.array(result)
      
    def __convert_to_k(self, target_int):
        match target_int:
                case 0:
                    return [1]+[0]*9
                case 1:
                    return [0]+[1]+[0]*8
                case 2:
                    return [0]*2+[1]+[0]*7
                case 3:
                    return [0]*3+[1]+[0]*6
                case 4:
                    return [0]*4+[1]+[0]*5
                case 5:
                    return [0]*5+[1]+[0]*4
                case 6:
                    return [0]*6+[1]+[0]*3
                case 7:
                    return [0]*7+[1]+[0]*2
                case 8:
                    return [0]*8+[1]+[0]*1
                case 9:
                    return [0]*9+[1]

class MetOfficeDataProvider(DataProvider):
    """South Scotland Met Office weather data provider."""

    def __init__(self, window_size, batch_size=10, max_num_batches=-1,
                 shuffle_order=True, rng=None):
        """Create a new Met Offfice data provider object.

        Args:
            window_size (int): Size of windows to split weather time series
               data into. The constructed input features will be the first
               `window_size - 1` entries in each window and the target outputs
               the last entry in each window.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """
        self.window_size = window_size
        assert window_size > 1, 'window_size must be at least 2.'
        data_path = os.path.join(
            os.environ['MLP_DATA_DIR'], 'HadSSP_daily_qc.txt')
        assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
        )
        # load raw data from text file
        path = "/afs/inf.ed.ac.uk/user/s19/s1953505/mlpractical/data/HadSSP_daily_qc.txt"
        raw_table = np.loadtxt(path, skiprows=3, usecols=[i+2 for i in range(31)])
        # filter out all missing datapoints and flatten to a vector
        filtered_vector = raw_table[raw_table != -99.99]
        # normalise data to zero mean, unit standard deviation
        minimum = min(filtered_vector)
        maximum = max(filtered_vector)
        d = maximum - minimum
        normalized_vector = [ (data-minimum)/d for data in filtered_vector]
        # convert from flat sequence to windowed data
        result = []
        current = [[]]
        for i in range(len(normalized_vector)):
            if ((i+1)%window_size==0):
                current.append(normalized_vector[i])
                result.append(current)
                current = [[]]
            else:
                current[0].append(normalized_vector[i])
        # inputs are first (window_size - 1) entries in windows
        inputs = np.array([[ inputs for inputs, _ in result]])
        # targets are last entry in windows
        targets = np.array([ target for _, target in result])
        # initialise base class with inputs and targets arrays
        super(MetOfficeDataProvider, self).__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng)
    def __next__(self):
            return self.next()