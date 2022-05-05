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

import sys
from datetime import datetime

import os

from app.argparser import get_common_argument_parser
from compression.configs.config import Config
from compression.data_loaders.creator import create_data_loader
from compression.engines.creator import create_engine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.optimization.optimizer_selector import OPTIMIZATION_ALGORITHMS
from compression.pipeline.initializer import create_pipeline
from compression.utils.logger import init_logger, get_logger

logger = get_logger(__name__)

_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def main():
    app(sys.argv[1:])


def app(argv):
    parser = get_common_argument_parser()
    args = parser.parse_args(args=argv)

    config = Config.read_config(args.config)
    config.update_from_args(args)

    if config.engine.type == 'simplified' and args.evaluate:
        raise Exception('Can not make evaluation in simplified mode')

    log_dir = _create_log_path(config)
    init_logger(level=args.log_level,
                file_name=os.path.join(log_dir, 'log.txt'),
                progress_bar=args.pbar)
    logger.info('Output log dir: {}'.format(log_dir))

    metrics = optimize(config)
    if metrics and logger.progress_bar_disabled:
        for name, value in metrics.items():
            logger.info('{: <27s}: {}'.format(name, value))


def _create_log_path(config):
    if config.model.direct_dump:
        model_log_dir = config.model.output_dir
        exec_log_dir = config.model.output_dir
    else:
        model_log_dir = os.path.join(config.model.output_dir, config.model.log_algo_name)
        exec_log_dir = os.path.join(model_log_dir, _timestamp)
    config.add_log_dir(model_log_dir, exec_log_dir)

    if not os.path.isdir(exec_log_dir):
        os.makedirs(exec_log_dir)

    return exec_log_dir


def print_optimizer_config(config):
    # log algorithms settings
    optimizer_string = 'Optimizer: {}'.format(config.name)
    optimizer_string += '\n Parameters:'
    for name, value in config.params.items():
        optimizer_string += '\n\t{: <27s}: {}'.format(name, value)
    optimizer_string += '\n {}'.format('=' * 75)
    logger.info(optimizer_string)


def print_algo_configs(config):
    # log algorithms settings
    configs_string = 'Creating pipeline:'
    for algo in config:
        configs_string += '\n Algorithm: {}'.format(algo.name)
        configs_string += '\n Parameters:'
        for name, value in algo.params.items():
            configs_string += '\n\t{: <27s}: {}'.format(name, value)
    configs_string += '\n {}'.format('=' * 75)
    logger.info(configs_string)


def optimize(config):
    """Creates pipeline of compression algorithms and optimize its parameters"""

    if logger.progress_bar_disabled:
        print_algo_configs(config.compression.algorithms)

    # load custom model
    model = load_model(config.model)

    data_loader = None
    # create custom data loader in case of custom Engine
    if config.engine.type != 'accuracy_checker':
        data_loader = create_data_loader(config.engine, model)

    engine = create_engine(config.engine, data_loader=data_loader, metric=None)

    pipeline = create_pipeline(
        config.compression.algorithms, engine)

    if 'optimizer' in config:
        print_optimizer_config(config.optimizer)
        optimizer = OPTIMIZATION_ALGORITHMS.get(config.optimizer.name)(config.optimizer, pipeline, engine)
        compressed_model = optimizer.run(model)
    else:
        compressed_model = pipeline.run(model)

    if not config.model.keep_uncompressed_weights:
        compress_model_weights(compressed_model)

    save_model(compressed_model,
               os.path.join(config.model.exec_log_dir, 'optimized'),
               model_name=config.model.model_name)

    # evaluating compressed model if need
    if config.engine.evaluate:
        return pipeline.evaluate(compressed_model)

    return None
