import scannerpy

class TestRealtimeKernel(scannerpy.Kernel):
  def __init__(self, config, protobufs):
    self.protobufs = protobufs
    pass

  def close(self):
    pass

  def execute(self, input_columns):
    return input_columns

KERNEL = TestRealtimeKernel
