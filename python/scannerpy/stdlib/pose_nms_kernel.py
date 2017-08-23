import scannerpy
import scannerpy.stdlib.parsers as parsers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.poses as poses

class PoseNMSKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        args = protobufs.PoseNMSKernelArgs()
        args.ParseFromString(config)
        self.height = args.height

    def close(self):
        pass

    def execute(self, input_columns):
        pose_list = []
        for c in input_columns:
            pose_list += parsers.poses(c, self.protobufs)
        nmsed_poses = poses.nms(pose_list, self.height * 0.2)
        return writers.poses([nmsed_poses], self.protobufs)

KERNEL = PoseNMSKernel
