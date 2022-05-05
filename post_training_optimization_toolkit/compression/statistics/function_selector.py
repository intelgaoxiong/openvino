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

from addict import Dict
from ..utils.registry import Registry

ACTIVATIONS = 'activations'
WEIGHTS = 'weights'

PERCHANNEL = 'perchannel'
PERTENSOR = 'pertensor'

AGGREGATION_FN = Registry('AggregationFunctions')

ACTIVATIONS_STATS_FN = Dict({
    PERCHANNEL: Registry('ActivationsPerchannelFunctions'),
    PERTENSOR: Registry('ActivationsPertensorFunctions')})

WEIGHTS_STATS_FN = Dict({
    PERCHANNEL: Registry('WeightsPerchannelFunctions'),
    PERTENSOR: Registry('WeightsPertensorFunctions')})


def get_aggregation_function(name):
    return AGGREGATION_FN.get(name)


def get_stats_function_for_activations(name, granularity):
    return ACTIVATIONS_STATS_FN[granularity].get(name)


def get_stats_function_for_weights(name, granularity):
    return WEIGHTS_STATS_FN[granularity].get(name)


def get_stats_function(tensor_type, name, granularity):
    if tensor_type == ACTIVATIONS:
        return get_stats_function_for_activations(name, granularity)
    if tensor_type == WEIGHTS:
        return get_stats_function_for_weights(name, granularity)
    raise RuntimeError('Type of tensor is not supported. Please use {} or {} types'.format(ACTIVATIONS, WEIGHTS))
