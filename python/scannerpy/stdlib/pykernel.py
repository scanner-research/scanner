import tensorflow as tf

class TensorFlowKernel:
    def __init__(self):
        # TODO: wrap this in "with device"
        self.graph = self.build_graph()
        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(config=config, graph=self.graph)

    def close(self):
        self.sess.close()

    def build_graph(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
