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

from copy import deepcopy

from ..magnitude_sparsity.algorithm import MagnitudeSparsity
from ...quantization.bias_correction.algorithm import BiasCorrection
from ...quantization.fast_bias_correction.algorithm import FastBiasCorrection
from ....algorithms.algorithm import Algorithm
from ....algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from ....statistics.collector import collect_statistics


@COMPRESSION_ALGORITHMS.register('BaseWeightSparsity')
class BaseWeightSparsity(Algorithm):
    name = 'BaseWeightSparsity'

    def __init__(self, config, engine, ignored_scope=None):
        super().__init__(config, engine)
        self.statistics_table = None
        bias_config = deepcopy(config)
        bias_config.threshold = 'inf'
        bias_config.apply_for_all_nodes = True
        use_fast_bias = self._config.get('use_fast_bias', True)
        bias_algo = FastBiasCorrection(bias_config, engine) if use_fast_bias \
            else BiasCorrection(bias_config, engine)
        self.algorithms = [MagnitudeSparsity(config, engine, ignored_scope=ignored_scope),
                           bias_algo]

    @property
    def change_original_model(self):
        return True

    def run(self, model):
        bias_correction = self.algorithms[1]
        collect_statistics(self._engine, model, [bias_correction])

        for algo in self.algorithms:
            algo.run(model)

        self.statistics_table = self.algorithms[0].statistics_table

        return model
