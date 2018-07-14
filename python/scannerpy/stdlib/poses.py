import numpy as np
import cv2
import copy
import math
from collections import defaultdict


class Pose(object):
    POSE_KEYPOINTS = 18
    POSE_SCORES = 1
    FACE_KEYPOINTS = 70
    HAND_KEYPOINTS = 21

    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

    # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/7325aa32dce312539e7414c1ba599631c3ad221b/include/openpose/pose/poseParametersRender.hpp
    DRAW_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
                  [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
                  [0, 14], [14, 16], [0, 15], [15, 17]]

    DRAW_COLORS = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [
        255, 255, 0
    ], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170],
                   [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
                   [255, 0, 170], [170, 0, 255], [255, 0, 255], [85, 0, 255]]

    def __init__(self):
        self.pose_score = 0
        self.keypoints = np.zeros((Pose.POSE_KEYPOINTS + Pose.FACE_KEYPOINTS +
                                   Pose.HAND_KEYPOINTS * 2, 3))

    def _format_keypoints(self):
        return self.keypoints

    def pose_keypoints(self):
        kp = self._format_keypoints()
        return kp[:self.POSE_KEYPOINTS, :]

    def face_keypoints(self):
        kp = self._format_keypoints()
        return kp[self.POSE_KEYPOINTS:(
            self.POSE_KEYPOINTS + self.FACE_KEYPOINTS), :]

    def hand_keypoints(self):
        kp = self._format_keypoints()
        base = kp[self.POSE_KEYPOINTS + self.FACE_KEYPOINTS:, :]
        return [base[:self.HAND_KEYPOINTS, :], base[self.HAND_KEYPOINTS:, :]]

    def face_bbox(self):
        p = self.pose_keypoints()

        # Keypoints for eyes, ears, and nose
        re = p[14, :]
        le = p[15, :]
        r = p[16, :]
        l = p[17, :]
        o = p[0, :]
        pts = [re, le, r, l, o]

        # Gather up all valid keypoints
        valid = []
        for pt in pts:
            if pt[2] > 0.05:
                valid.append(pt)

        if len(valid) == 0:
            return [(0, 0), (0, 0), 0]

        face = np.array(valid, ndmin=2)

        # Find the x extent of the keypoints
        xmin = face[:, 0].min()
        xmax = face[:, 0].max()
        width = xmax - xmin
        xmin -= width * 0.1
        xmax += width * 0.1

        yavg = np.mean(face[:, 1])

        ymin = yavg - width
        ymax = yavg + width

        score = min(p[16, 2], p[17, 2], p[0, 2])
        return [(xmin, ymin), (xmax, ymax), score]

    def body_bbox(self):
        p = self.pose_keypoints()
        xmin = p[:, 0].min()
        xmax = p[:, 0].max()
        ymin = p[:, 1].min()
        ymax = p[:, 1].max()
        score = np.mean(p[:, 2])
        return [(xmin, ymin), (xmax, ymax), score]

    def draw(self, img, thickness=5, draw_threshold=0.05):
        def to_pt(i):
            def check(v):
                return v >= 0 and v < 1 and v == v
            if not check(self.keypoints[i, 0]) or not check(self.keypoints[i, 1]):
                return None
            return (int(self.keypoints[i, 0] * img.shape[1]),
                    int(self.keypoints[i, 1] * img.shape[0]))

        for ([a, b], color) in zip(self.DRAW_PAIRS, self.DRAW_COLORS):
            if self.keypoints[a, 2] > draw_threshold and \
               self.keypoints[b, 2] > draw_threshold:
                pt_a = to_pt(a)
                pt_b = to_pt(b)
                if pt_a is not None and pt_b is not None:
                    cv2.line(img, pt_a, pt_b, color, thickness)

        return img

    def distance_to(self, pose, confidence_threshold=0.2):
        kp = self.pose_keypoints()
        other_kp = pose.pose_keypoints()

        distances = []
        for pi in range(self.POSE_KEYPOINTS):
            if (kp[pi, 2] > confidence_threshold and
                other_kp[pi, 2] > confidence_threshold):
                dist = math.sqrt((other_kp[pi, 0] - kp[pi, 0])**2 +
                                 (other_kp[pi, 1] - kp[pi, 1])**2)
                distances.append(dist)

        if len(distances) == 0:
            score = float('inf')
        else:
            score = np.median(distances)

        return score

    @staticmethod
    def from_buffer(keypoints_buffer):
        pose = Pose()
        shape = pose.keypoints.shape
        data = np.frombuffer(keypoints_buffer, dtype=np.float32)
        pose.pose_score = data[0]
        pose.keypoints = data[1:].reshape(shape)
        return pose


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
    pose_scores = np.sum(joints_4d[:, 2, :], axis=0)
    num_joints_per_pose = np.sum(joints_4d[:, 2, :] > 0.2, axis=0)
    # sort by score
    idxs = np.argsort(pose_scores)
    idxs_orig = np.argsort(pose_scores)

    # spatially hash joints into buckets
    x_buckets = [defaultdict(set) for _ in range(num_joints)]
    y_buckets = [defaultdict(set) for _ in range(num_joints)]
    for i, idx in enumerate(idxs):
        pose = poses[idx]
        for pi in range(num_joints):
            if pose[pi, 2] > 0.2:
                x_pos = int(pose[pi, 1] - (pose[pi, 1] % overlapThresh))
                y_pos = int(pose[pi, 0] - (pose[pi, 0] % overlapThresh))
                for xp in range(x_pos - 1, x_pos + 2):
                    x_buckets[pi][xp].add(idx)
                for yp in range(y_pos - 1, y_pos + 2):
                    y_buckets[pi][yp].add(idx)

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
            if pose[pi, 2] > 0.2:
                x_pos = int(pose[pi, 1] - (pose[pi, 1] % overlapThresh))
                y_pos = int(pose[pi, 0] - (pose[pi, 0] % overlapThresh))

                x_set = set()
                for xp in range(x_pos - 1, x_pos + 2):
                    x_set.update(x_buckets[pi][xp])
                y_set = set()
                for yp in range(y_pos - 1, y_pos + 2):
                    y_set.update(y_buckets[pi][yp])
                both_set = x_set.intersection(y_set)
                # Increment num overlaps for each joint
                for idx in both_set:
                    overlaps[idx] += 1

        duplicates = []
        for idx, num_overlaps in overlaps.items():
            if num_overlaps >= min(3, num_joints_per_pose[idx]):
                for ii, idx2 in enumerate(idxs):
                    if idx == idx2:
                        break
                duplicates.append(ii)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.array(duplicates))))

    # return only the bounding boxes that were picked
    out_poses = []
    for i in pick:
        out_poses.append(orig_poses[i])
    return out_poses
