import scannerpy
import struct

class MyOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        input_count = len(input_columns[0])
        column_count = len(input_columns)
        return [[struct.pack('=q', 9000) for _ in xrange(input_count)] 
                 for _ in xrange(column_count)]

KERNEL = MyOpKernel
