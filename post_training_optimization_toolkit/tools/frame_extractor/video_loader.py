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

from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT


class VideoLoader:

    def __init__(self, video_file):
        self._frame_pointer = -1

        self._video_file = video_file
        self._capture = VideoCapture(video_file)

    def __getitem__(self, idx):

        if idx >= len(self) or idx < 0:
            raise IndexError

        if idx < self._frame_pointer:
            self._capture = VideoCapture(self._video_file)
            self._frame_pointer = -1

        image = None
        while self._frame_pointer < idx:
            _, image = self._capture.read()
            self._frame_pointer += 1

        if image is not None:
            return image

    def __len__(self):
        return int(self._capture.get(CAP_PROP_FRAME_COUNT))
