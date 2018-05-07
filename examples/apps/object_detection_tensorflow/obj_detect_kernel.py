
import numpy as np
import tensorflow as tf
import cv2
import os
import scannerpy

from scannerpy.stdlib import tensorflow
from typing import Tuple

##################################################################################################
# Assume that DNN model is located in PATH_TO_GRAPH with filename 'frozen_inference_graph.pb'    #
# Example model can be downloaded from:                                                          #
# http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz #
##################################################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_REPO = script_dir

PATH_TO_GRAPH = os.path.join(PATH_TO_REPO, 'ssd_mobilenet_v1_coco_2017_11_17',
                             'frozen_inference_graph.pb')


@scannerpy.register_python_op()
class ObjDetect(tensorflow.TensorFlowKernel):
    def build_graph(self):
        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return dnn

    # Evaluate object detection DNN model on a frame
    # Return bounding box position, class and score
    def execute(self, frame: FrameType) -> bytes:
        image = frame
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        scores = self.graph.get_tensor_by_name('detection_scores:0')
        classes = self.graph.get_tensor_by_name('detection_classes:0')
        with self.graph.as_default():
            (boxes, scores, classes) = self.sess.run(
                [boxes, scores, classes],
                feed_dict={
                    image_tensor: np.expand_dims(image, axis=0)
                })

            # bundled data format: [box position(x1 y1 x2 y2), box class, box score]
            bundled_data = np.concatenate(
                (boxes.reshape(100, 4), classes.reshape(100, 1),
                 scores.reshape(100, 1)), 1)[:20]

            return bundled_data.tobytes()
