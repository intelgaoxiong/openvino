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
from argparse import ArgumentParser

import extractor


def parse_args(argv):
    """
    Parse and process arguments for frames-extractor tool
    """
    parser = ArgumentParser(description='Frames-extractor toolkit', allow_abbrev=False)
    parser.add_argument(
        '-v',
        '--video',
        help='Full path to video file',
        required=True)
    parser.add_argument(
        '-o',
        '--output_dir',
        help='Directory to save valuable frames from video.',
        required=True)
    parser.add_argument(
        '-f',
        '--frame_step',
        type=int,
        help='Read frames from video with step',
        default=1,
        required=False
    )
    parser.add_argument(
        '-e',
        '--ext',
        type=str,
        help='Extension of images in resulting dataset',
        choices=['jpg', 'png'],
        default='png',
        required=False
    )
    parser.add_argument(
        '-s',
        '--dataset_size',
        type=int,
        help='Number of frames to save from video as dataset.'
             'Should be less then video frames number',
        default=None,
        required=False)
    args = parser.parse_args(args=argv)

    return args.video, args.output_dir, args.dataset_size, args.frame_step


if __name__ == '__main__':
    extractor.extract_frames_and_make_dataset(*parse_args(sys.argv[1:]))
