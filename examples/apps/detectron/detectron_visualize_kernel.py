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

import utils.vis as vis_utils
import datasets.dummy_datasets as dummy_datasets
import pickle

from scannerpy import Kernel

class DetectronVizualizeKernel(Kernel):
    def __init__(self, config, protobufs):
        self.dataset = dummy_datasets.get_coco_dataset()
        pass

    def execute(self, cols):
        print('in viz')
        print(globals())
        image = cols[0]
        cls_boxes = pickle.loads(cols[1])
        cls_segms = pickle.loads(cols[2])
        cls_keyps = pickle.loads(cols[3])

        vis_im = vis_utils.vis_one_image_opencv(
            image[:, :, ::1],  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=self.dataset,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

        return [vis_im]

KERNEL = DetectronVizualizeKernel
