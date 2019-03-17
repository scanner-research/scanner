from ..kernel import Kernel
from scannerpy import DeviceType


class TensorFlowKernel(Kernel):
    def __init__(self, config):
        import tensorflow as tf

        # If this is a CPU kernel, tell TF that it should not use
        # any GPUs for its graph operations
        cpu_only = True
        visible_device_list = []
        tf_config = tf.ConfigProto()
        for handle in config.devices:
            if handle.type == DeviceType.GPU.value:
                visible_device_list.append(str(handle.id))
                tf_config.gpu_options.allow_growth = True
                cpu_only = False
        if cpu_only:
            tf_config.device_count['GPU'] = 0
        else:
            tf_config.gpu_options.visible_device_list = ','.join(visible_device_list)
        # TODO: wrap this in "with device"
        self.config = config
        self.tf_config = tf_config

    def close(self):
        self.sess.close()

    def setup_with_resources(self):
        import tensorflow as tf

        self.graph = self.build_graph()
        self.sess = tf.Session(config=self.tf_config, graph=self.graph)
        self.sess.as_default()

    def build_graph(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
