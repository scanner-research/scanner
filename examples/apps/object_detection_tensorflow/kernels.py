import numpy as np
import tensorflow as tf
import cv2
import os
import scannerpy
import scannerpy.stdlib.util
import pickle
import visualization_utils as vis_util
import tarfile

from scannerpy import FrameType, DeviceType
from scannerpy.stdlib import tensorflow
from typing import Tuple
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
PATH_TO_REPO = script_dir

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(PATH_TO_REPO, 'data', 'mscoco_label_map.pbtxt')

categories = vis_util.parse_labelmap(PATH_TO_LABELS)
category_index = vis_util.create_category_index(categories)

def download_and_extract_model(url):
    path = scannerpy.stdlib.util.download_temp_file(url)
    tar_file = tarfile.open(path)
    for f in tar_file.getmembers():
        file_name = os.path.basename(f.name)
        if 'frozen_inference_graph.pb' in file_name:
            local_path = scannerpy.stdlib.util.temp_directory()
            tar_file.extract(f, local_path)
            model_path = os.path.join(local_path, f.name)
            break
    return model_path

@scannerpy.register_python_op()
class ObjDetect(tensorflow.TensorFlowKernel):
    def build_graph(self):
        url = self.config.args['dnn_url']
        # Download the DNN model
        path = download_and_extract_model(url)

        dnn = tf.Graph()
        with dnn.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
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

##################################################################################################
# Driver Functions                                                                               #
##################################################################################################

# Intersection Over Union (Area)
def IoU(box1, box2):
    # intersection rectangle (y1, x1, y2, x2)
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    area_intersection = (x2 - x1) * (y2 - y1)

    area_box1 = (box1[3] - box1[1]) * (box1[2] - box1[0])
    area_box2 = (box2[3] - box2[1]) * (box2[2] - box2[0])

    area_union = area_box1 + area_box2 - area_intersection

    return area_intersection * 1.0 / area_union


# non-maximum suppression
def nms_single(bundled_data, iou_threshold=0.5):
    bundled_data = bundled_data.reshape(20, 6)
    data_size = len(bundled_data)
    repeated_indices = []
    selected_indices = set(range(data_size))

    [boxes, classes, scores] = np.split(bundled_data, [4, 5], axis=1)

    for i in range(data_size):
        for j in range(i + 1, data_size):
            if IoU(boxes[i],
                   boxes[j]) > iou_threshold and classes[i] == classes[j]:
                repeated_indices.append(j)

    repeated_indices = set(repeated_indices)
    selected_indices = list(selected_indices - repeated_indices)

    selected_bundled_data = np.take(bundled_data, selected_indices, axis=0)
    [boxes_np, classes_np, scores_np] = np.split(
        selected_bundled_data, [4, 5], axis=1)

    return [boxes_np, classes_np, scores_np]


# tried to use multiprocessing module to scale,
# but doesn't work well because of high overhead cost
def nms_bulk(bundled_data_list):
    print("Working on non-maximum suppression...")
    bundled_np_list = [
        nms_single(bundled_data) for bundled_data in tqdm(bundled_data_list)
    ]
    print("Finished non-maximum suppression!")
    return bundled_np_list


def neighbor_boxes(box1, box2, threshold=0.1):
    r"""This method returns whether two boxes are close enough.
    If two boxes from two neighboring frames are considered
    close enough, they are refered as the same object.
    """

    if math.abs(box1[0] - box2[0]) > threshold:
        return False
    if math.abs(box1[1] - box2[1]) > threshold:
        return False
    if math.abs(box1[2] - box2[2]) > threshold:
        return False
    if math.abs(box1[3] - box2[3]) > threshold:
        return False
    return True


def smooth_box(bundled_np_list, min_score_thresh=0.5):
    r"""If you knew which boxes in consecutive frames were the same object,
    you could "smooth" out of the box positions over time.
    For example, the box position at frame t could be the average of
    the positions in surrounding frames:
    box_t = (box_(t-1) + box_t + box_(t+1)) / 3
    """
    print("Working on making boxes smooth...")

    for i, now in enumerate(tqdm(bundled_np_list)):
        # Ignore the first and last frame
        if i == 0:
            before = None
            continue
        else:
            before = bundled_np_list[i - 1]

        if i == len(bundled_np_list) - 1:
            after = None
            continue
        else:
            after = bundled_np_list[i + 1]

        [boxes_after, classes_after, scores_after] = after
        [boxes_before, classes_before, scores_before] = before
        [boxes_now, classes_now, scores_now] = now

        for j, [box_now, class_now, score_now] in enumerate(
                zip(boxes_now, classes_now, scores_now)):

            # Assume that the boxes list is already sorted
            if score_now < min_score_thresh:
                break

            confirmed_box_after = None
            confirmed_box_before = None

            for k, [box_after, class_after, score_after] in enumerate(
                    zip(boxes_after, classes_after, scores_after)):
                if (IoU(box_now, box_after) > 0.3 and
                    class_now == class_after and
                    score_after > min_score_thresh - 0.1):

                    confirmed_box_after = box_after
                    if score_after < min_score_thresh:
                        scores_after[k] = score_now
                    break

            for k, [box_before, class_before, score_before] in enumerate(
                    zip(boxes_before, classes_before, scores_before)):
                if IoU(box_now,
                       box_before) > 0.3 and class_now == class_before:
                    confirmed_box_before = box_before
                    break

            if confirmed_box_before is not None and confirmed_box_after is not None:
                box_now += box_now
                box_now += confirmed_box_before
                box_now += confirmed_box_after
                box_now /= 4.0
            elif confirmed_box_before is not None:
                box_now += confirmed_box_before
                box_now /= 2.0
            elif confirmed_box_after is not None:
                box_now += confirmed_box_after
                box_now /= 2.0

            boxes_now[j] = box_now

        bundled_np_list[i] = [boxes_now, classes_now, scores_now]
        bundled_np_list[i + 1] = [boxes_after, classes_after, scores_after]
        bundled_np_list[i - 1] = [boxes_before, classes_before, scores_before]

    print("Finished making boxes smooth!")
    return bundled_np_list


@scannerpy.register_python_op(name='TFDrawBoxes')
def draw_boxes(config, frame: FrameType, bundled_data: bytes) -> FrameType:
    min_score_thresh = config.args['min_score_thresh']
    [boxes, classes, scores] = pickle.loads(bundled_data)
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=min_score_thresh)
    return frame
