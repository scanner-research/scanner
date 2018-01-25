# Mostly taken from: https://github.com/tensorflow/models/blob/1f34fc/research/object_detection/object_detection_tutorial.ipynb

import numpy as np
import tensorflow as tf
import cv2
import os
from scannerpy.stdlib import pykernel
from utils import visualization_utils as vis_util

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_REPO = script_dir

##################################################################################################
# Assume that DNN model is located in PATH_TO_GRAPH with filename 'frozen_inference_graph.pb'    #
# Example model can be downloaded from:                                                          #
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz #
##################################################################################################

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TO_REPO, 'data', 'mscoco_label_map.pbtxt')

PATH_TO_GRAPH = os.path.join(PATH_TO_REPO, 'ssd_mobilenet_v1_coco_2017_11_17', 'frozen_inference_graph.pb')

categories = vis_util.parse_labelmap(PATH_TO_LABELS)
category_index = vis_util.create_category_index(categories)

class Kernel(pykernel.TensorFlowKernel):
    def build_graph(self):
        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return dnn

    def execute(self, cols):
        print 'Execute'
        image = cols[0]
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        (boxes, scores, classes) = self.sess.run(
            [boxes, scores, classes],
            feed_dict={image_tensor: np.expand_dims(image, axis=0)})
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return [image.tobytes()]
