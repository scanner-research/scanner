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

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pickle
import numpy as np

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
import scannerpy
import pycocotools.mask as mask_util

from scannerpy import Kernel, FrameType, DeviceType
from scannerpy.stdlib import caffe2
from typing import Tuple, Sequence


@scannerpy.register_python_op(device_type=DeviceType.GPU)
class Detectron(caffe2.Caffe2Kernel):
    def build_graph(self):
        c2_utils.import_detectron_ops()
        # OpenCL may be enabled by default in OpenCV3; disable it because it's not
        # thread safe and causes unwanted GPU memory allocations.
        cv2.ocl.setUseOpenCL(False)

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

    def execute(self, frame: FrameType) -> Tuple[bytes, bytes, bytes]:
        logger = logging.getLogger(__name__)

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.graph, frame, None, timers=timers)
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        return (
            pickle.dumps(cls_boxes),
            pickle.dumps(cls_segms),
            pickle.dumps(cls_keyps))


@scannerpy.register_python_op(name='DetectronVizualize')
def detectron_vizualize(config,
                        frame: FrameType,
                        cls_boxes: bytes,
                        cls_segms: bytes,
                        cls_keyps: bytes) -> FrameType:
        cls_boxes = pickle.loads(cls_boxes)
        cls_segms = pickle.loads(cls_segms)
        cls_keyps = pickle.loads(cls_keyps)

        vis_im = vis_utils.vis_one_image_opencv(
            frame[:, :, ::1],  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_datasets.get_coco_dataset(),
            show_class=True,
            thresh=0.7,
            kp_thresh=2)

        return vis_im


@scannerpy.register_python_op(name='MaskNonBBox')
def mask_non_bbox(config, frame: FrameType, cls_segms: bytes) -> FrameType:
    cls_segms = pickle.loads(cls_segms)
    _, segms, _, _ = vis_utils.convert_from_cls_format([], cls_segms, None)

    if segms is not None and len(segms) > 0:
        masks = mask_util.decode(segms)

    sum_mask = np.zeros_like(frame)[..., 0]
    for mi in range(masks.shape[2]):
        sum_mask = np.logical_or(sum_mask, masks[:, :, mi])

    idx = np.nonzero(np.invert(sum_mask))
    img = frame.copy()

    img[idx[0], idx[1], :] = 0

    return img
