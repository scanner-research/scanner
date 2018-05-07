from ..kernel import Kernel
from scannerpy import DeviceType

from caffe2.python import workspace


class Caffe2Kernel(Kernel):
    def __init__(self, config):
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        self.config = config
        self.protobufs = config.protobufs
        self.graph = self.build_graph()

    def close(self):
        del self.graph

    def build_graph(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
