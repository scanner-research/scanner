import scannerpy
import scannerpy.stdlib.writers as writers


class TestPyKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        assert (config.args['kernel_arg'] == 1)
        pass

    def close(self):
        pass

    def execute(self, input_columns):
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        return [point.SerializeToString()]


KERNEL = TestPyKernel
