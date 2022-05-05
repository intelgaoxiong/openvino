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

import os
import tempfile


def create_tmp_dir(parent_dir=tempfile.gettempdir()):
    """ Creates temporary directory with unique name and auto cleanup
    :param parent_dir: directory in which temporary directory is created
    :return: TemporaryDirectory object
    """
    parent_dir = tempfile.TemporaryDirectory(dir=parent_dir)
    if not os.path.exists(parent_dir.name):
        try:
            os.makedirs(parent_dir.name)
        except PermissionError as e:
            raise type(e)(
                'Failed to create directory {}. Permission denied. '.format(parent_dir))
    return parent_dir


def convert_output_key(name):
    """ Convert output name into IE-like name
    :param name: output name to convert
    :return: IE-like output name
    """
    if not isinstance(name, tuple):
        return name
    if len(name) != 2:
        raise Exception('stats name should be a string name or 2 elements tuple '
                        'with string as the first item and port number the second')
    return '{}.{}'.format(*name)


class Environment:
    def __init__(self, variable, value):
        self.variable = variable
        self.value = value
        self.was_set = (self.variable in os.environ)

    def __enter__(self):
        os.environ[self.variable] = self.value

    def __exit__(self, *args):
        if not self.was_set:
            del os.environ[self.variable]
