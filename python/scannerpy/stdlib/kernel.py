from __future__ import absolute_import, division, print_function, unicode_literals
from ..kernel import Kernel

import tensorflow as tf

class TensorFlowKernel(Kernel):
    def __init__(self, config, protobufs):
        # TODO: wrap this in "with device"
        self.config = config
        self.tf_config = tf.ConfigProto(allow_soft_placement = True)
        self.graph = self.build_graph()
        self.sess = tf.Session(graph=self.graph, config=self.tf_config)
        self.sess.as_default()
        self.protobufs = protobufs

    def close(self):
        self.sess.close()

    def build_graph(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
