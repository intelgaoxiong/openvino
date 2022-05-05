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

from ..utils.registry import Registry, RegistryStorage

COMPRESSION_ALGORITHMS = Registry('QuantizationAlgos')
REGISTRY_STORAGE = RegistryStorage(globals())


def get_registry(name):
    return REGISTRY_STORAGE.get_registry(name)


def get_algorithm(name):
    if name.startswith('.') or name.endswith('.'):
        raise Exception('The algorithm name cannot start or end with "."')

    if '.' in name:
        ind = name.find('.')
        reg_name = name[:ind]
        algo_name = name[ind + 1:]
    else:
        reg_name = 'QuantizationAlgos'
        algo_name = name

    reg = get_registry(reg_name)
    return reg.get(algo_name)
