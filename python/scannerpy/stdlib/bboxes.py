from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import cv2

from scannerpy.table import Table
import scannerpy.stdlib.parsers


def proto_to_np(bboxes):
    return [[
        box.x1, box.y1, box.x2, box.y2, box.score, box.track_id,
        box.track_score
    ] for box in bboxes]


def nms(orig_boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(orig_boxes) == 0:
        return []
    elif len(orig_boxes) == 1:
        return orig_boxes

    if type(orig_boxes[0]) != np.ndarray:
        boxes = proto_to_np(orig_boxes)
    else:
        boxes = orig_boxes

    npboxes = np.array(boxes[0])
    for box in boxes[1:]:
        npboxes = np.vstack((npboxes, box))
    boxes = npboxes
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs,
                         np.concatenate(
                             ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return np.array(orig_boxes)[pick]


def draw(vid_table,
         bbox_table,
         output_path,
         fps=24,
         threshold=0.0,
         color=(255, 0, 0)):
    if isinstance(bbox_table, Table):
        rows = bbox_table.parent_rows()
        bboxes = [b for _, b in bbox_table.load([0], parsers.bboxes)]
    else:
        [rows, bboxes] = zip(*bbox_table)
    frames = [f[0] for _, f in vid_table.load([0], rows=rows)]

    frame_shape = frames[0].shape
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'X264'), fps,
                             (frame_shape[1], frame_shape[0]))

    for (frame, frame_bboxes) in zip(frames, bboxes):
        for bbox in frame_bboxes:
            if bbox.score < threshold: continue
            cv2.rectangle(frame, (int(bbox.x1), int(bbox.y1)),
                          (int(bbox.x2), int(bbox.y2)), color, 3)
        output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
