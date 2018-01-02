import scannerpy
import scannerpy.stdlib.writers as writers

class TestPyFailKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        pass

    def close(self):
        pass

    def execute(self, input_columns):
        raise scannerpy.ScannerException('Test')

KERNEL = TestPyFailKernel
