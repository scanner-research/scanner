import numpy as np

from scannerpy.table import Table


def proto_to_np(bboxes):
    return np.array([[
        box.x1, box.y1, box.x2, box.y2, box.score, box.label, box.track_id,
        box.track_score
    ] for box in bboxes])

# Frome https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(bbox_a, bbox_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox_a.x1, bbox_b.x1)
    yA = max(bbox_a.y1, bbox_b.y1)
    xB = min(bbox_a.x2, bbox_b.x2)
    yB = min(bbox_a.y2, bbox_b.y2)

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bbox_a.x2 - bbox_a.x1 + 1) * (bbox_a.y2 - bbox_a.y1 + 1)
    boxBArea = (bbox_b.x2 - bbox_b.x1 + 1) * (bbox_b.y2 - bbox_b.y1 + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

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
