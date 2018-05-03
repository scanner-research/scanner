# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pickle

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

from scannerpy.stdlib import caffe2

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


class DetectronKernel(caffe2.Caffe2Kernel):
    def build_graph(self):
        merge_cfg_from_file(self.config.args['config_path'])

        # If this is a CPU kernel, tell Caffe2 that it should not use
        # any GPUs for its graph operations
        cpu_only = True
        for handle in self.config.devices:
            if handle.type == DeviceType.GPU.value:
                cpu_only = False

        if cpu_only:
            cfg.NUM_GPUS = 0
        else:
            cfg.NUM_GPUS = 1
        # TODO: wrap this in "with device"
        weights_path = cache_url(self.config.args['weights_path'],
                                 cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg(cache_urls=False)
        model = infer_engine.initialize_model_from_cfg(weights_path)
        return model

    def execute(self, cols):
        print(len(cols))
        logger = logging.getLogger(__name__)

        image = cols[0]

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.graph, image, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        return [pickle.dumps(cls_boxes),
                pickle.dumps(cls_segms),
                pickle.dumps(cls_keyps)]


KERNEL = DetectronKernel
