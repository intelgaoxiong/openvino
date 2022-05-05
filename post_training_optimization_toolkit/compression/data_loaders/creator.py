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

from compression.data_loaders.image_loader import ImageLoader
from compression.graph.model_utils import get_nodes_by_type


def create_data_loader(config, model):
    """
    Factory to create instance of engine class based on config
    :param config: engine config section from toolkit config file
    :param model: NXModel instance to find out input shape
    :return: instance of DataLoader descendant class
    """

    inputs = get_nodes_by_type(model, ['Parameter'])

    if len(inputs) > 1 and\
            not any([tuple(i.shape) == (1, 3) for i in inputs]):
        raise RuntimeError('IEEngine supports networks with single input or net with 2 inputs. '
                           'In second case there are image input and image info input '
                           'Actual inputs number: {}'.format(len(inputs)))

    data_loader = None
    for in_node in inputs:
        if tuple(in_node.shape) != (1, 3):
            data_loader = ImageLoader(config)
            data_loader.shape = in_node.shape
            return data_loader

    if data_loader is None:
        raise RuntimeError('There is no node with image input')

    return data_loader
