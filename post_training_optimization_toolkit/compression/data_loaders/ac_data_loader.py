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

from typing import Union

from ..api.data_loader import DataLoader
from ..utils.ac_imports import Dataset, DatasetWrapper


class ACDataLoader(DataLoader):

    def __init__(self, data_loader: Union[Dataset, DatasetWrapper]):
        super().__init__(config=None)
        self._data_loader = data_loader

    def __len__(self):
        return self._data_loader.full_size

    def __getitem__(self, item):
        return self._data_loader[item]
