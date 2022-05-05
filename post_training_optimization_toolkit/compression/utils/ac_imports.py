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


try:
    from libs.open_model_zoo.tools.accuracy_checker.\
        accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
    from libs.open_model_zoo.tools.accuracy_checker.accuracy_checker.config import ConfigReader
    from libs.open_model_zoo.tools.accuracy_checker.accuracy_checker.dataset import\
        Dataset, DataProvider as DatasetWrapper
    from libs.open_model_zoo.tools.accuracy_checker.accuracy_checker.logging\
        import _DEFAULT_LOGGER_NAME

except ImportError:
    from accuracy_checker.evaluators.quantization_model_evaluator import create_model_evaluator
    from accuracy_checker.config import ConfigReader
    from accuracy_checker.dataset import Dataset
    from accuracy_checker.logging import _DEFAULT_LOGGER_NAME
    try:
        from accuracy_checker.dataset import DataProvider as DatasetWrapper
    except ImportError:
        from accuracy_checker.dataset import DatasetWrapper
