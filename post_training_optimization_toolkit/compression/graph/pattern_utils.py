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

from .pattern_builder import PatternBuilder
from .node_utils import get_node_input, get_all_node_outputs
from ..graph.special_operations import OPERATIONS_WITH_BIAS, ELTWISE_TYPES, is_eltwise


def check_fused_scale_shift_patterns(match):
    node = [match[node] for node in match if match[node].kind == 'op' and match[node].type == 'Multiply'][0]
    return check_for_branching(node)


def get_fused_scale_shift_patterns():
    return [pattern.insert_scaleshift().pattern for pattern in fc_conv_fused_basis_patterns()]


def check_fused_op_const_patterns(match):
    nodes = [match[node] for node in match if match[node].kind == 'op' and is_eltwise(match[node])]
    return any([check_for_branching(node) for node in nodes])


def get_fused_op_const_pattern():
    return [pattern.append_op_const(lambda x: x in ELTWISE_TYPES, 'eltwise').pattern
            for pattern in fc_conv_fused_basis_patterns()]


def fc_conv_fused_basis_patterns():
    base_pattern = PatternBuilder().insert_conv_fc(name='input')

    pattern_with_bias = PatternBuilder()
    pattern_with_bias.insert_conv_fc(name='input')
    pattern_with_bias.insert_bias()

    pattern_with_activation = PatternBuilder()
    pattern_with_activation.insert_conv_fc(name='input')
    pattern_with_activation.insert_activation()

    pattern_with_both = PatternBuilder()
    pattern_with_both.insert_conv_fc(name='input')
    pattern_with_both.insert_bias()
    pattern_with_both.insert_activation()

    return [
        base_pattern,
        pattern_with_bias,
        pattern_with_activation,
        pattern_with_both
    ]


def check_for_branching(node):
    parent_node_type = None
    # This list of types ends propagation through constant and FC nodes
    skip_types = [op['type'] for op in OPERATIONS_WITH_BIAS]
    skip_types.append('Const')
    while parent_node_type not in skip_types:
        parent_node = get_node_input(node, 0)
        parent_node_type = parent_node.type
        if len(get_all_node_outputs(parent_node)) > 1:
            return True
        node = parent_node
    return False
