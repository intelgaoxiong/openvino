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

import random

from compression.engines.ac_engine import ACEngine
from compression.samplers.batch_sampler import BatchSampler
from compression.samplers.index_sampler import IndexSampler


def create_sampler(engine, samples, shuffle_data=False, seed=0):
    """ Helper function to create the most common samplers. Suits for the most algorithms
    :param engine: instance of engine class
    :param samples: a list of dataset indices or a number of samples to draw from dataset
    :param shuffle_data: a boolean flag. If it's True and samples param is a number then
     subset indices will be choice randomly
    :param seed: a number for initialization of the random number generator
    :return instance of Sampler class suitable to passed engine
    """

    if isinstance(samples, int):
        if shuffle_data:
            random.seed(seed)
            samples = random.sample(range(len(engine.data_loader)), samples)
        else:
            samples = range(samples)

    if isinstance(engine, ACEngine):
        return IndexSampler(subset_indices=samples)

    return BatchSampler(engine.data_loader, subset_indices=samples)
