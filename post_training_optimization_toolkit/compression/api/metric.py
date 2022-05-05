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

from abc import ABC, abstractmethod


class Metric(ABC):
    """An abstract class representing an accuracy metric. """

    def __init__(self):
        """ Constructor """
        self.reset()

    @property
    @abstractmethod
    def value(self):
        """ Returns accuracy metric value for the last model output. """

    @property
    @abstractmethod
    def avg_value(self):
        """ Returns accuracy metric value for all model outputs. """

    @abstractmethod
    def update(self, output, target):
        """ Calculates and updates accuracy metric value
        :param output: model output
        :param target: annotations
        """

    @abstractmethod
    def reset(self):
        """ Reset accuracy metric """

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    @property
    def higher_better(self):
        """Attribute whether the metric should be increased"""
        return True

    @abstractmethod
    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
