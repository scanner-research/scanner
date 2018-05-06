import scannerpy
import scannerpy.stdlib.readers as readers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.poses as poses


@scannerpy.register_python_op(
    variadic_inputs=True,
    outputs=['pose'])
class PoseNMSKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        self.height = config.args['height']

    def close(self):
        pass

    def execute(self, input_columns):
        pose_list = []
        for c in input_columns:
            pose_list += readers.poses(c, self.protobufs)
        nmsed_poses = poses.nms(pose_list, self.height * 0.2)
        return [writers.poses(nmsed_poses, self.protobufs)]
