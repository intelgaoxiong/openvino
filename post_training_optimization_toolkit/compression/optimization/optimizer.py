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

from ..api.engine import Engine
from ..pipeline.pipeline import Pipeline


class Optimizer(ABC):
    def __init__(self, config, pipeline: Pipeline, engine: Engine):
        """ Constructor
         :param config: optimizer config
         :param pipeline: pipeline of algorithms to optimize
         :param engine: entity responsible for communication with dataset
          """
        self._config, self._pipeline, self._engine = config.params, pipeline, engine
        self.name = config.name

    @abstractmethod
    def run(self, model):
        """ Run optimizer on model
        :param model: model to apply optimization
         """
