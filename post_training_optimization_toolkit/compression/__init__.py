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

from .algorithms.quantization.accuracy_aware.algorithm import AccuracyAwareQuantization
from .algorithms.quantization.accuracy_aware.mixed_precision import (
    INT4MixedQuantization,
)
from .algorithms.quantization.fast_bias_correction.algorithm import FastBiasCorrection
from .algorithms.quantization.bias_correction.algorithm import BiasCorrection
from .algorithms.quantization.channel_alignment.algorithm import (
    ActivationChannelAlignment,
)
from .algorithms.quantization.datafree.algorithm import DataFreeQuantization
from .algorithms.quantization.default.algorithm import DefaultQuantization
from .algorithms.quantization.minmax.algorithm import MinMaxQuantization
from .algorithms.quantization.optimization.rangeopt import RangeOptimization
from .algorithms.quantization.optimization.params_tuning import (
    ParamsGridSearchAlgorithm,
)
from .algorithms.quantization.qnoise_estimator.algorithm import QuantNoiseEstimator
from .algorithms.quantization.tunable_quantization.algorithm import TunableQuantization
from .algorithms.quantization.outlier_channel_splitting.algorithm import (
    OutlierChannelSplitting,
)
from .algorithms.quantization.weight_bias_correction.algorithm import (
    WeightBiasCorrection,
)
from .algorithms.sparsity.magnitude_sparsity.algorithm import MagnitudeSparsity
from .algorithms.sparsity.default.algorithm import WeightSparsity
from .algorithms.sparsity.default.base_algorithm import BaseWeightSparsity
from .optimization.tpe.base_algorithm import Tpe

QUANTIZATION_ALGORITHMS = [
    'MinMaxQuantization',
    'RangeOptimization',
    'FastBiasCorrection',
    'BiasCorrection',
    'ActivationChannelAlignment',
    'DataFreeQuantization',
    'DefaultQuantization',
    'AccuracyAwareQuantization',
    'INT4MixedQuantization',
    'TunableQuantization',
    'Tpe',
    'QuantNoiseEstimator',
    'OutlierChannelSplitting',
    'WeightBiasCorrection',
    'ParamsGridSearchAlgorithm',
]

SPARSITY_ALGORITHMS = ['WeightSparsity',
                       'MagnitudeSparsity',
                       'BaseWeightSparsity']

__all__ = QUANTIZATION_ALGORITHMS + SPARSITY_ALGORITHMS
