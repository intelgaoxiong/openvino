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

from .pattern_utils import check_fused_scale_shift_patterns, get_fused_scale_shift_patterns, \
    check_fused_op_const_patterns, get_fused_op_const_pattern


def get_cpu_ignored_patterns():
    return {
        'blocks': [(pattern, check_fused_scale_shift_patterns) for pattern in get_fused_scale_shift_patterns()] +
                  [(pattern, check_fused_op_const_patterns) for pattern in get_fused_op_const_pattern()],
        'activations': [],
        'inputs': []
    }
