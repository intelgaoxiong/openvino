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

import os
import json
import cv2
import numpy as np

from compression.api import DataLoader


class COCOLoader(DataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.images_path = config.images_path
        self.annotation_path = config.annotation_path
        self.images = os.listdir(self.images_path)
        self.labels = None
        self.data, self.bbox = self.prepare_annotation()

    def prepare_annotation(self):
        with open(self.annotation_path) as f:
            file = json.load(f)
        self.labels = [i['id'] for i in file['categories']]
        data = {}
        for idx, image in enumerate(file['images']):
            data[idx] = {'file_name': image['file_name'], 'image_id': image['id'],
                         'height': image['height'], 'width': image['width']}
        bbox = {}
        for i in file['annotations']:
            if i['image_id'] not in bbox.keys():
                bbox[i['image_id']] = {'bbox': [self.prepare_bbox(*i['bbox'])],
                                       'category_id': [i['category_id']],
                                       'iscrowd': [i['iscrowd']]}
            else:
                bbox[i['image_id']]['bbox'].append(self.prepare_bbox(*i['bbox']))
                bbox[i['image_id']]['category_id'].append(i['category_id'])
                bbox[i['image_id']]['iscrowd'].append(i['iscrowd'])

        return data, bbox

    def __getitem__(self, index):
        """ Returns (img_id, img_annotation), image"""
        if index >= len(self):
            raise IndexError

        shape_image = self.data[index]['height'], self.data[index]['width']

        bbox = np.array(self.bbox.get(self.data[index]['image_id'], {}).get('bbox', []))

        if bbox.size != 0:
            x_maxs = np.max(bbox, axis=1)
            y_maxs = np.max(bbox, axis=1)
            x_mins = np.min(bbox, axis=1)
            y_mins = np.min(bbox, axis=1)
        else:
            x_maxs, y_maxs, x_mins, y_mins = [], [], [], []

        labels = np.array(self.bbox.get(self.data[index]['image_id'], {}).get('category_id', []))
        iscrowd = np.array(self.bbox.get(self.data[index]['image_id'], {}).get('iscrowd', []))

        annotation = {'boxes': bbox, 'labels': labels, 'iscrowd': iscrowd,
                      'x_maxs': x_maxs, 'x_mins': x_mins, 'y_maxs': y_maxs, 'y_mins': y_mins}
        annotation = (index, [annotation, shape_image])
        return annotation, self._read_and_preprocess_image(self.images_path + self.data[index]['file_name'])

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _read_and_preprocess_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (640, 640))
        return image.transpose(2, 0, 1)

    @staticmethod
    def prepare_bbox(x, y, weight, height):
        return np.array([x, y, x + weight, y + height])
