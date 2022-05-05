#
# Copyright 2021 Intel Corporation.
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

try:
    # pylint: disable=unused-import
    import hyperopt
    from .algorithm import TpeOptimizer

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False


from compression.optimization.optimizer import Optimizer
from compression.optimization.optimizer_selector import OPTIMIZATION_ALGORITHMS


@OPTIMIZATION_ALGORITHMS.register('Tpe')
class Tpe(Optimizer):
    def __init__(self, config, pipeline, engine):
        super().__init__(config, pipeline, engine)
        if HYPEROPT_AVAILABLE:
            self.optimizer = TpeOptimizer(config, pipeline, engine)
        else:
            raise ModuleNotFoundError(
                'Cannot import the hyperopt package which is a dependency '
                'of the TPE algorithm. '
                'Please install hyperopt via `pip install hyperopt==0.1.2 pandas==0.24.2`'
            )

    def run(self, model):
        return self.optimizer.run(model)
