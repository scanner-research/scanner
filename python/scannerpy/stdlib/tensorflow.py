from ..kernel import Kernel
from scannerpy import DeviceType

import tensorflow as tf


class TensorFlowKernel(Kernel):
    def __init__(self, config, protobufs):
        # If this is a CPU kernel, tell TF that it should not use
        # any GPUs for its graph operations
        cpu_only = True
        for handle in config.devices:
            if handle.type == DeviceType.GPU.value:
                cpu_only = False
        if cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.ConfigProto(allow_soft_placement=True)
        # TODO: wrap this in "with device"
        self.config = config
        self.tf_config = tf_config
        self.graph = self.build_graph()
        self.sess = tf.Session(config=self.tf_config, graph=self.graph)
        self.sess.as_default()
        self.protobufs = protobufs

    def close(self):
        self.sess.close()

    def build_graph(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
