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

from compression.engines.ac_engine import ACEngine
from compression.engines.simplified_engine import SimplifiedEngine


def create_engine(config, **kwargs):
    """
    Factory to create instance of engine class based on config
    :param config: engine config section from toolkit config file
    :param kwargs: additional arguments specific for every engine class (data_loader, metric)
    :return: instance of Engine descendant class
    """
    if config.type == 'accuracy_checker':
        return ACEngine(config)
    if config.type == 'simplified':
        return SimplifiedEngine(config, **kwargs)
    raise RuntimeError('Unsupported engine type')
