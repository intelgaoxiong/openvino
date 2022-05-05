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

from abc import ABC, abstractmethod
from copy import deepcopy

from ..api.engine import Engine


class Algorithm(ABC):

    algo_type = 'quantization'

    def __init__(self, config, engine: Engine):
        """ Constructor
         :param config: algorithm specific config
         :param engine: model inference engine
         :param sampler: Sampler class inheritor instance to read dataset
          """
        self._config, self._engine = deepcopy(config), engine
        self._stats_collector = None
        self.params = {}
        self.default_steps_size = 0.05
        self.total_exec_steps = 0

    @property
    def config(self):
        return self._config

    @property
    def algo_collector(self):
        return self._stats_collector

    @algo_collector.setter
    def algo_collector(self, collector):
        self._stats_collector = collector

    @abstractmethod
    def run(self, model):
        """ Run algorithm on model
        :param model: model to apply algorithm
        :return optimized model
         """

    def statistics(self):
        """ Returns a dictionary of printable statistics"""
        return {}

    def register_statistics(self, model, stats_collector):
        """
        :param model: FP32 original model
        :param stats_collector: object of StatisticsCollector class
        :return: None
        """

    def get_parameter_meta(self, _model, _optimizer_state):
        """ Get parameters metadata
        :param _model: model to get parameters for
        :param _optimizer_state: dictionary describing optimizer state that allow to tune created search space
        differently for different optimizer states
        :return params_meta: metadata of optional parameters
        """
        return []

    def compute_total_exec_steps(self, model=None):
        """ Compute executions steps based on stat_subset_size, algorithm, model """

    def update_config(self, config):
        """ Update Algorithm configuration based on input config """
        self._config = deepcopy(config)
