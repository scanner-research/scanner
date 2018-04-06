import scannerpy
import scannerpy.stdlib.writers as writers
import pickle

class TestPyKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        assert (config.args['kernel_arg'] == 1)
        self.x = 20
        self.y = 20

    def close(self):
        pass

    def new_stream(self, args):
        if args is None:
            return
        if 'x' in args:
            self.x = args['x']
        if 'y' in args:
            self.y = args['y']

    def execute(self, input_columns):
        point = {}
        point['x'] = self.x
        point['y'] = self.y
        return [pickle.dumps(point)]


KERNEL = TestPyKernel
