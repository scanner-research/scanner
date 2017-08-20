import scannerpy.stdlib.writers as writers

class TestPyKernel:
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        pass

    def close(self):
        pass

    def execute(self, input_columns):
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        return [point.SerializeToString()]

KERNEL = TestPyKernel
