from ..table import Table
import numpy as np
import cv2
import parsers
import copy
from collections import defaultdict

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

    num_joints = poses[0].shape[0]

    max_boxes = len(poses)
    joints_4d = np.stack(poses, axis=2)
    pose_scores = np.sum(joints_4d[:,2,:], axis=0)
    num_joints_per_pose = np.sum(joints_4d[:,2,:] > 0.2, axis=0)
    # sort by score
    idxs = np.argsort(pose_scores)
    idxs_orig = np.argsort(pose_scores)

    # spatially hash joints into buckets
    x_buckets = [defaultdict(set) for _ in range(num_joints)]
    y_buckets = [defaultdict(set) for _ in range(num_joints)]
    for i, idx in enumerate(idxs):
        pose = poses[idx]
        for pi in range(num_joints):
            if pose[pi,2] > 0.2:
                x_pos = pose[pi,1] - (pose[pi,1] % overlapThresh)
                y_pos = pose[pi,0] - (pose[pi,0] % overlapThresh)
                x_buckets[pi][x_pos].add(idx)
                y_buckets[pi][y_pos].add(idx)

    # the list of picked indexes
    pick = []

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        overlaps = defaultdict(int)
        pose = poses[i]
        for pi in range(num_joints):
            if pose[pi,2] > 0.2:
                x_pos = pose[pi,1] - (pose[pi,1] % overlapThresh)
                y_pos = pose[pi,0] - (pose[pi,0] % overlapThresh)

                x_set = x_buckets[pi][x_pos]
                y_set = y_buckets[pi][y_pos]
                both_set = x_set.intersection(y_set)
                # Increment num overlaps for each joint
                for idx in both_set:
                    overlaps[idx] += 1

        duplicates = []
        for idx, num_overlaps in overlaps.iteritems():
            if num_overlaps >= min(3, num_joints_per_pose[idx]):
                for ii, idx2 in enumerate(idxs):
                    if idx == idx2:
                        break
                duplicates.append(ii)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.array(duplicates))))

    # return only the bounding boxes that were picked
    out_poses = []
    for i in pick:
        out_poses.append(orig_poses[i])
    return out_poses
