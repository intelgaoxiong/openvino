#
# Copyright 2020-2021 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version October 2018)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.

from abc import ABC, abstractmethod


class Sampler(ABC):

    def __init__(self, data_loader=None, batch_size=1, subset_indices=None):
        """ Constructor
        :param data_loader: instance of DataLoader class to read data
        :param batch_size: number of items in batch
        :param subset_indices: indices of samples to read
        If subset_indices argument is set to None then Sampler class
        will take elements from the whole dataset"""
        self._data_loader, self.batch_size = data_loader, batch_size
        self._subset_indices = subset_indices

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @property
    def num_samples(self):
        if self._subset_indices is None:
            return len(self._data_loader) if self._data_loader else None
        return len(self._subset_indices)
