#
# Copyright 2021 Intel Corporation.
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

import torch

from compression.graph import node_utils as nu
from compression.utils.logger import get_logger


logger = get_logger(__name__)


def get_optimization_params(loss_name, optimizer_name):
    loss_fn_map = {
        'l2': torch.nn.MSELoss(),
    }

    optimizer_map = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
    }
    return loss_fn_map[loss_name], optimizer_map[optimizer_name]


def get_weight_node(node, port_id=1):
    node_weight = nu.get_node_input(node, port_id)
    if node_weight.type == 'FakeQuantize':
        node_weight = nu.get_node_input(node_weight, 0)
    if node_weight.type != 'Const':
        raise ValueError('Provided weight node is not Const!')
    return node_weight
