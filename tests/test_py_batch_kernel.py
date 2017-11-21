import scannerpy
import scannerpy.stdlib.writers as writers

class TestPyBatchKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        pass

    def close(self):
        pass

    def execute(self, input_columns):
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        input_count = len(input_columns[0])
        column_count = len(input_columns)
        return [[point.SerializeToString() for _ in xrange(input_count)]
                 for _ in xrange(column_count)]

KERNEL = TestPyBatchKernel
