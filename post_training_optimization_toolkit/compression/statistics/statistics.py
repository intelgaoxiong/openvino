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

from functools import partial


class Statistic:
    def __init__(self, func, *argv, **kwargs):
        self.func = func
        self.argv = argv
        self.kwargs = kwargs

    def compute(self, *input_tensor, **kwargs):
        pass

    def __eq__(self, other):
        if isinstance(other, Statistic):
            return self.func == other.func and self.argv == other.argv \
                   and self.kwargs == other.kwargs
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __call__(self, *argv, **kwargs):
        return self.compute(*argv, **kwargs)

    def __hash__(self):
        data = (self.func, frozenset(self.argv), frozenset(self.kwargs))
        if isinstance(self.func, partial):
            data = (*data, frozenset(self.func.keywords))
        return hash(data)


class TensorStatistic(Statistic):
    def compute(self, *input_tensor, **kwargs):
        return self.func(*(input_tensor + self.argv), **self.kwargs)


class SQNRStatistic(Statistic):
    def __init__(self, func, qsuffix, *argv, **kwargs):
        super().__init__(func, *argv, **kwargs)
        self.qsuffix = qsuffix

    # pylint: disable=W0221
    def compute(self, activation_dict, layer_key, **kwargs):
        return self.func(
            activation_dict[layer_key],
            activation_dict[layer_key + self.qsuffix],
            **kwargs
        )


def compute_statistic(statistic, *argv, **kwargs):
    if isinstance(argv[0], dict):
        activations_dict, layer_key = argv
        tensor = (activations_dict[layer_key],)
        if isinstance(statistic, SQNRStatistic):
            return statistic.compute(activations_dict, layer_key)
    else:
        tensor = argv
    return statistic(*tensor, **kwargs)
