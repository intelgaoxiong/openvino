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


class DataLoader(ABC):
    """An abstract class representing a dataset.

    All custom datasets should inherit.
    ``__len__`` provides the size of the dataset and
    ``__getitem__`` supports integer indexing in range from 0 to len(self)
    """

    def __init__(self, config):
        """ Constructor
        :param config: data loader specific config
        """
        self.config = config

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass
