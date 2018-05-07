import scannerpy
import scannerpy.stdlib.readers as readers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.poses as poses

from typing import Tuple

@scannerpy.register_python_op()
class PoseNMSKernel(scannerpy.Kernel):
    def __init__(self, config):
        self.protobufs = config.protobufs
        self.height = config.args['height']

    def close(self):
        pass

    def execute(self, *inputs : Tuple[bytes]) -> bytes:
        pose_list = []
        for c in inputs:
            pose_list += readers.poses(c, self.protobufs)
        nmsed_poses = poses.nms(pose_list, self.height * 0.2)
        return writers.poses(nmsed_poses, self.protobufs)
