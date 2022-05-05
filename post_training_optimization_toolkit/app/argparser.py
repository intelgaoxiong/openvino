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

from argparse import ArgumentParser


def get_common_argument_parser():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser(description='Post-training Compression Toolkit', allow_abbrev=False)

    parser.add_argument(
        '-c',
        '--config',
        help='Path to a config file with task/model-specific parameters',
        required=True)

    parser.add_argument(
        '-e',
        '--evaluate',
        action='store_true',
        help='Whether to evaluate model on whole dataset')

    # Storage settings
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='The directory where models are saved. Default: ./results')

    parser.add_argument(
        '-d',
        '--direct-dump',
        action='store_true',
        help='Flag to save files without sub folders with algo names')

    log_levels = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=log_levels,
        help='Log level to print')

    parser.add_argument(
        '--progress-bar',
        dest='pbar',
        action='store_true',
        default=False,
        help='Disable CL logging and enable progress bar')

    parser.add_argument(
        '--stream-output',
        dest='stream_output',
        action='store_true',
        default=False,
        help='Switch progress display to a multiline mode')

    parser.add_argument(
        '--keep-uncompressed-weights',
        action='store_true',
        default=False,
        help='Keep Convolution, Deconvolution and FullyConnected weights uncompressed')

    return parser
