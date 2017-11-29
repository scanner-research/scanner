from __future__ import absolute_import, division, print_function, unicode_literals

class KernelConfig(object):
    def __init__(self, device_handles, input_columns, input_column_types,
                 output_columns, args, node_id):
        self.devices = device_handles
        self.input_columns = input_columns
        self.input_column_types = input_column_types
        self.output_columns = output_columns
        self.args = args
        self.node_id = node_id


class Kernel(object):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        pass
