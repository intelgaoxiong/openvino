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

import glob
import os.path
import importlib
import inspect
import mo.ops
import extensions.ops


def get_operations_list():
    ops = {}
    for package in [extensions.ops, mo.ops]:
        for file in glob.glob(os.path.join(os.path.dirname(os.path.abspath(package.__file__)), '*.py')):
            name = '.{}'.format(os.path.splitext(os.path.basename(file))[0])
            module = importlib.import_module(name, package.__name__)
            for key in dir(module):
                obj = getattr(module, key)
                if inspect.isclass(obj):
                    op = getattr(obj, 'op', None)
                    if op is not None:
                        ops[op] = obj
    return ops


OPERATIONS = get_operations_list()
