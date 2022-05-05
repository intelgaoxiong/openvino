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

from collections import defaultdict
import numpy as np

from ..statistics.statistics import compute_statistic
from ..utils.logger import get_logger

logger = get_logger(__name__)


def append_stats(accumulated_layer_stats, stats_layout, value, dataset_index):
    if isinstance(value, list):
        value = parse_sequential_stats(value, stats_layout)

    for layer in value:
        if layer in stats_layout:
            if layer not in accumulated_layer_stats:
                accumulated_layer_stats[layer] = {stat_name: [] for stat_name in stats_layout[layer]}
            for stat_name, stat_fn in stats_layout[layer].items():
                accumulated_layer_stats[layer][stat_name].append(
                    (dataset_index, compute_statistic(stat_fn, value, layer)))


def parse_sequential_stats(value_sequential, stats_layout):
    activation_seq = defaultdict(lambda: [])
    for value in value_sequential:
        for layer, activations in value.items():
            if layer in stats_layout:
                activation_seq[layer].append(activations)

    for layer, act_seq in activation_seq.items():
        activation_seq[layer] = np.stack(act_seq, axis=2)
    return activation_seq


def process_accumulated_stats(accumulated_stats, stat_names_aliases=None):
    for layer in accumulated_stats:
        for stat in accumulated_stats[layer]:
            accumulated_stats[layer][stat].sort(key=lambda el: el[0])
            accumulated_stats[layer][stat] = [el[1] for el in accumulated_stats[layer][stat]]

    # pack IE-like stats names into original ones
    if stat_names_aliases is not None:
        accumulated_stats = {stat_names_aliases[key]: value
                             for key, value in accumulated_stats.items()}

    return accumulated_stats
