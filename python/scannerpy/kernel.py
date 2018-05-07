import pickle


class KernelConfig(object):
    def __init__(self, config):
        self.devices = config.devices
        self.input_columns = config.input_columns
        self.input_column_types = config.input_column_types
        self.output_columns = config.output_columns
        self.output_column_types = config.output_column_types
        self.args = pickle.loads(config.args())
        self.node_id = config.node_id


class Kernel(object):
    def __init__(self, config):
        self.config = config
        self.protobufs = config.protobufs

    def close(self):
        pass

    def new_stream(self, args):
        pass

    def reset(self):
        pass

    def execute(self, input_columns):
        raise NotImplementedError
