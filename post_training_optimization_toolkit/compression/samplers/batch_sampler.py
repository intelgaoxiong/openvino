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

from compression.samplers.sampler import Sampler


class BatchSampler(Sampler):

    def __init__(self, data_loader, batch_size=1, subset_indices=None):
        super().__init__(data_loader, batch_size, subset_indices)
        if self._subset_indices is None:
            self._subset_indices = range(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch = []
        for idx in self._subset_indices:
            batch.append(self._data_loader[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
