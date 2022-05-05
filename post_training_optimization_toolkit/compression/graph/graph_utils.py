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
from copy import deepcopy

from mo.graph.graph import Graph
from mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph
from mo.utils.logger import init_logger

from ..graph.passes import ModelPreprocessor
from ..utils.logger import stdout_redirect

init_logger('ERROR', False)


def load_graph(model_config):
    """ Loads model from specified path
    :return NetworkX model
     """
    bin_path = model_config.weights
    xml_path = model_config.model

    if not os.path.exists(xml_path):
        raise RuntimeError('Input model xml should link to an existing file. Please, provide a correct path.')

    if not os.path.exists(bin_path):
        raise RuntimeError('Input model bin should link to an existing file. Please, provide a correct path.')

    graph_from_ir, meta_data = stdout_redirect(restore_graph_from_ir, xml_path, bin_path)

    meta_data['quantization_parameters'] = model_config.quantization_info
    graph_from_ir.meta_data = meta_data
    graph_from_ir.ir_v10 = True
    model_preprocessing(graph_from_ir)
    return graph_from_ir


def save_graph(graph: Graph, save_path, model_name=None):
    """ Save model as IR in specified path
    :param graph: NetworkX model to save
    :param save_path: path to save the model
    :param model_name: name under which the model will be saved
     """
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except PermissionError as e:
            raise type(e)(
                'Failed to create a directory {}. Permission denied. '.format(save_path))
    else:
        if not os.access(save_path, os.W_OK):
            raise PermissionError(
                'Output directory {} is not writable for the current user. '.format(save_path))

    save_restored_graph(graph=deepcopy(graph), path=save_path, meta_data=graph.meta_data,
                        name=model_name)


def model_preprocessing(model):
    ModelPreprocessor().find_and_replace_pattern(model)
    model.clean_up()
