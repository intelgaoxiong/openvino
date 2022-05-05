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

from addict import Dict
from .pipeline import Pipeline
from ..algorithms.algorithm_selector import get_algorithm


def create_pipeline(algo_config, engine):
    """ Create instance of Pipeline class from config file and add specified algorithms
    :param algo_config: list of algorithms configurations
    :param engine: engine to use for inference
    :return: instance of Pipeline class
    """
    pipeline = Pipeline(engine)

    for algo in algo_config:
        if not isinstance(algo, Dict):
            algo = Dict(algo)
        algo_type = algo.name
        algo_params = algo.params
        pipeline.add_algo(get_algorithm(algo_type)(algo_params, engine))

    return pipeline
