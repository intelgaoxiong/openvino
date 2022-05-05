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
    import jstyleson as json
except ImportError:
    import json
from pathlib import Path
import yaml


def read_config_from_file(path):
    path = Path(path)
    extension = path.suffix.lower()
    with path.open() as f:
        if extension in ('.yaml', '.yml'):
            return yaml.load(f, Loader=yaml.SafeLoader)
        if extension in ('.json',):
            return json.load(f)
        raise RuntimeError('Unknown file extension for the file "{}"'.format(path))


def write_config_to_file(data, path):
    path = Path(path)
    extension = path.suffix.lower()
    with path.open('w') as f:
        if extension in ('.yaml', '.yml'):
            yaml.dump(data, f)
        elif extension in ('.json',):
            json.dump(data, f)
        else:
            raise RuntimeError('Unknown file extension for the file "{}"'.format(path))
