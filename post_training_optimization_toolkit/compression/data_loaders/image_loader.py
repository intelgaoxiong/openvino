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

from cv2 import imread, IMREAD_GRAYSCALE

from ..api.data_loader import DataLoader
from ..data_loaders.utils import prepare_image, collect_img_files


class ImageLoader(DataLoader):

    def __init__(self, config):
        super().__init__(config)

        self._img_files = collect_img_files(config.data_source)
        self._shape = None
        self._crop_central_fraction = config.get('central_fraction', None)

    def __getitem__(self, idx):
        return self._read_and_preproc_image(self._img_files[idx])

    def __len__(self):
        return len(self._img_files)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = tuple(shape)

    def _read_and_preproc_image(self, img_path):
        image = imread(img_path, IMREAD_GRAYSCALE)\
            if self._shape[1] == 1 else imread(img_path)

        if image is None:
            raise Exception('Can not read the image: {}'.format(img_path))

        return prepare_image(image, self.shape[-2:], self._crop_central_fraction)
