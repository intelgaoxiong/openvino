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

QUANTIZE_AGNOSTIC_OPERATIONS = [
    {'type': 'MaxPool'},
    {'type': 'ReduceMax'},
    {'type': 'Reshape'},
    {'type': 'Concat'},
    {'type': 'Flatten'},
    {'type': 'Squeeze'},
    {'type': 'Unsqueeze'},
    {'type': 'Split'},
    {'type': 'VariadicSplit'},
    {'type': 'Crop'},
    {'type': 'Transpose'},
    {'type': 'Tile'},
    {'type': 'StridedSlice'},
    {'type': 'ShuffleChannels'},
    {'type': 'Broadcast'},
    {'type': 'Pad'},
]

OPERATIONS_WITH_BIAS = [
    {'type': 'Convolution'},
    {'type': 'MatMul'}
]

OPERATIONS_WITH_WEIGHTS = [
    {'type': 'Convolution'},
    {'type': 'ConvolutionBackpropData'},
    {'type': 'GroupConvolution'},
    {'type': 'GroupConvolutionBackpropData'},
    {'type': 'MatMul'},
]

SPLIT_OPERATIONS = [
    {'type': 'VariadicSplit'},
    {'type': 'Split'}
]

DETECTION_OUTPUT_FINAL_TYPES = [
    {'type': 'NonMaxSuppression'},
    {'type': 'TopK'}
]

ELTWISE_TYPES = ['Add', 'Multiply', 'Subtract', 'Divide', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
                 'Equal', 'NotEqual', 'FloorMod', 'LogicalOr', 'LogicalXor', 'LogicalAnd', 'Maximum', 'Minimum']


def is_eltwise(node):
    return node.type in ELTWISE_TYPES
