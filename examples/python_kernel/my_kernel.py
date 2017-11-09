import scannerpy
import struct

class MyOpKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        print('list size :{:d}'.format(len(input_columns)))
        return [struct.pack('=q', 9000)]

KERNEL = MyOpKernel
