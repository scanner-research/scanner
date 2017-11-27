import cv2

import scannerpy
from scannerpy.stdlib import parsers

class PoseDrawKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        frame = input_columns[0]
        frame_poses = input_columns[1]
        for all_pose in parsers.poses(frame_poses, self.protobufs):
            pose = all_pose.pose_keypoints()
            for i in range(18):
                if pose[i, 2] < 0.35: continue
                print(pose[i, 1], pose[i, 0])
                print(frame.shape)
                x = int(pose[i, 0] * frame.shape[1])
                y = int(pose[i, 1] * frame.shape[0])
                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (255, 0, 0), 3)
        return [frame]

KERNEL = PoseDrawKernel
