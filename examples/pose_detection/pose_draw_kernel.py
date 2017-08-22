import cv2

class PoseDrawKernel:
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        frame = input_columns[0]
        frame_poses = input_columns[1]
        for pose in parsers.poses(frame_poses, self.protobufs):
            for i in range(18):
                if pose[i, 2] < 0.35: continue
                cv2.circle(
                    frame,
                    (int(pose[i, 1]), int(pose[i, 0])),
                    8,
                    (255, 0, 0), 3)
        return [frame]

KERNEL = PoseDrawKernel
