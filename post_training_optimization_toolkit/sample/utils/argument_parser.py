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


def get_common_argparser():
    parser = ArgumentParser(description='Post-training Compression Toolkit Sample',
                            allow_abbrev=False)
    parser.add_argument(
        '-m',
        '--model',
        help='Path to the xml file with model',
        required=True)
    parser.add_argument(
        '-w',
        '--weights',
        help='Path to the bin file with model weights',
        required=False)
    parser.add_argument(
        '-d',
        '--dataset',
        help='Path to the directory with images',
        required=True)

    return parser
