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

from compression import MinMaxQuantization
from compression.algorithms.algorithm_selector import COMPRESSION_ALGORITHMS
from compression.algorithms.quantization import fake_quantize as fqut
from compression.algorithms.quantization import utils as ut
from compression.algorithms.quantization.fake_quantize import insert_fake_quantize_nodes, compute_stats_layouts
from compression.algorithms.quantization.fake_quantize_configuration import read_all_fake_quantize_configurations
from compression.algorithms.quantization.utils import load_hardware_config
from compression.graph.model_utils import get_nodes_by_type
from compression.graph.node_utils import get_node_input


@COMPRESSION_ALGORITHMS.register('TunableQuantization')
class TunableQuantization(MinMaxQuantization):
    name = 'TunableQuantization'

    @property
    def change_original_model(self):
        return False

    def run(self, model):
        """ this function applies tunable quantization algorithm
         :param model: model to apply algo
         :return model with inserted and filled FakeQuantize nodes
         """
        activation_statistics = self._stats_collector.get_statistics_for_algorithm(self.name)
        return fqut.get_quantized_model(model,
                                        self.create_stats_layout,
                                        activation_statistics,
                                        self.fill_fq_range,
                                        self._config,
                                        self.params)

    def register_statistics(self, model, stats_collector):
        model = deepcopy(model)
        insert_fake_quantize_nodes(self._config, model, self.params)
        activation_statistics_layout = self.__get_activations_statistics_layout(model, qscheme=self.params)
        stats_collector.register(self.name, activation_statistics_layout, self._sampler)
        self._stats_collector = stats_collector

    def __get_activations_statistics_layout(self, model, qscheme=None):
        """
        Compute statistics layout for activations
        :param model: NXModel instance
        :return: statistics layout in format {node_name: [stat_1, stat_2] .. }
        """
        fake_quantize_config = compute_stats_layouts(self._config, model, qscheme=qscheme)

        activations_stats_layout = self.create_stats_layout(fake_quantize_config, model, for_weights=False)

        return activations_stats_layout

    def get_parameter_meta(self, model, optimizer_state):
        param_grid = []
        if 'range_estimator' in self._config.tuning_scope:
            for variable in self._config.estimator_tuning_scope:
                self._config.tuning_scope.append('estimator_' + variable)
        config = deepcopy(self._config)
        if optimizer_state['first_iteration'] or optimizer_state['fully_quantized']:
            config['tuning_scope'] = []

        hardware_config = load_hardware_config(config)
        model = deepcopy(model)
        insert_fake_quantize_nodes(config, model)
        fq_configuration = read_all_fake_quantize_configurations(config, hardware_config, model)

        nodes_config = {}
        for fq in get_nodes_by_type(model, ['FakeQuantize']):
            node_input = get_node_input(fq, 0)
            op_type = 'weights' if node_input.type == 'Const' else 'activations'
            fq_node_config = fq_configuration[fq.name][op_type]
            for child_name, child_config in fq_node_config:
                if child_name not in nodes_config:
                    nodes_config[child_name] = {'weights': [], 'activations': []}
                nodes_config[child_name][op_type].extend(child_config)

        for node_name, node_config in nodes_config.items():
            if 'activations' in node_config:
                node_config['activations'] = ut.append_estimator_configs(
                    node_config['activations'], False, config,
                    self.params[node_name] if not optimizer_state['fully_quantized']
                    and node_name in self.params else None)
            if 'weights' in node_config:
                node_config['weights'] = ut.append_estimator_configs(
                    node_config['weights'], True, config,
                    self.params[node_name] if not optimizer_state['fully_quantized']
                    and node_name in self.params else None)

        for node_name, node_config in nodes_config.items():
            op_config = ut.get_quantize_op_config(node_config, config,
                                                  self.params[node_name] if not optimizer_state['fully_quantized']
                                                  and node_name in self.params else None)
            param_grid.append((node_name, 'choice', op_config))
        return param_grid
