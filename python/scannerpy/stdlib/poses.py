from ..table import Table
import numpy as np
import cv2
import parsers
import copy

def scale_pose(pose, scale):
    new_pose = pose.copy()
    for i in range(15):
        new_pose[i] *= scale
    return new_pose

def nms(orig_poses, overlapThresh):
    # if there are no boxes, return an empty list
    if len(orig_poses) == 0:
        return []
    elif len(orig_poses) == 1:
        return orig_poses

    poses = copy.deepcopy(orig_poses)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions

    # initialize the list of picked indexes
    pick = []

    # Keypoints: nose, neck, right eye, left eye, right ear, left ear
    head_joints = [0, 1, 14, 15, 16, 17]
    # compute the convex hull of the head
    max_boxes = len(poses)
    x1 = np.zeros(shape=(max_boxes))
    y1 = np.zeros(shape=(max_boxes))
    x2 = np.zeros(shape=(max_boxes))
    y2 = np.zeros(shape=(max_boxes))
    score = np.zeros(shape=(max_boxes))
    num_valid = 0
    for pi in range(len(poses)):
        temp_x1 = 100000.0
        temp_y1 = 100000.0
        temp_x2 = -100000.0
        temp_y2 = -100000.0
        temp_score = 0.0
        pose = poses[pi]
        for j in head_joints:
            s = pose[j,2]
            if s > 0.2:
                temp_x1 = min(temp_x1, pose[j,1])
                temp_y1 = min(temp_y1, pose[j,0])
                temp_x2 = max(temp_x2, pose[j,1])
                temp_y2 = max(temp_y2, pose[j,0])
                temp_score = max(temp_score, s)
        if temp_x1 > temp_x2 or temp_y1 > temp_y2:
            continue
        x1[num_valid] = temp_x1
        y1[num_valid] = temp_y1
        x2[num_valid] = temp_x2
        y2[num_valid] = temp_y2
        score[num_valid] = temp_score

        num_valid += 1

    x1.resize((num_valid))
    y1.resize((num_valid))
    x2.resize((num_valid))
    y2.resize((num_valid))
    score.resize((num_valid))
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
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    out_poses = []
    for i in pick:
        out_poses.append(orig_poses[i])
    return out_poses
