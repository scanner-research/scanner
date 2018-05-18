import numpy as np
import cv2
import copy
from collections import defaultdict


class Pose(object):
    POSE_KEYPOINTS = 18
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
        l = p[16, :2]
        r = p[17, :2]
        o = p[0, :2]
        up = o + [r[1] - l[1], l[0] - r[0]]
        down = o + [l[1] - r[1], r[0] - l[0]]
        face = np.array([l, r, up, down])

        xmin = face[:, 0].min()
        xmax = face[:, 0].max()
        ymin = face[:, 1].min()
        ymax = face[:, 1].max()

        score = min(p[16, 2], p[17, 2], p[0, 2])
        return [(xmin, ymin), (xmax, ymax), score]

    def draw(self, img, thickness=5, draw_threshold=0.05):
        def to_pt(i):
            return (int(self.keypoints[i, 0] * img.shape[1]),
                    int(self.keypoints[i, 1] * img.shape[0]))

        for ([a, b], color) in zip(self.DRAW_PAIRS, self.DRAW_COLORS):
            if self.keypoints[a, 2] > draw_threshold and \
               self.keypoints[b, 2] > draw_threshold:
                cv2.line(img, to_pt(a), to_pt(b), color, thickness)

        return img

    @staticmethod
    def from_buffer(keypoints_buffer):
        pose = Pose()
        shape = pose.keypoints.shape
        pose.keypoints = (np.frombuffer(keypoints_buffer,
                                        dtype=np.float32).reshape(shape))
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
