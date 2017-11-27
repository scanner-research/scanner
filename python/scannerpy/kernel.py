from __future__ import absolute_import, division, print_function, unicode_literals

class Kernel(object):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        pass
