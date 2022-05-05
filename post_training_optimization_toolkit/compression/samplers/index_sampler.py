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


class IndexSampler(Sampler):

    def __init__(self, subset_indices):
        super().__init__(subset_indices=subset_indices)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for idx in self._subset_indices:
            yield idx

    def __getitem__(self, idx):
        return self._subset_indices[idx]
