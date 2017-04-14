import logging as log
import numpy as np
import enum
from collections import defaultdict


class ScannerException(Exception):
    pass


class DeviceType(enum.Enum):
    """ Enum for specifying where an Op should run. """
    CPU = 0
    GPU = 1

    @staticmethod
    def to_proto(db, device):
        if device == DeviceType.CPU:
            return db.protobufs.CPU
        elif device == DeviceType.GPU:
            return db.protobufs.GPU
        else:
            raise ScannerException('Invalid device type')


class Job:
    def __init__(self, columns, name=None):
        self._columns = columns
        self._name = name

    def name(self):
        return self._name

    def op(self, db):
        return db.ops.Output(inputs=self._columns)
