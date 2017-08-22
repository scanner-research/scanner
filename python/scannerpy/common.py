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
    def to_proto(protobufs, device):
        if device == DeviceType.CPU:
            return protobufs.CPU
        elif device == DeviceType.GPU:
            return protobufs.GPU
        else:
            raise ScannerException('Invalid device type')


class ColumnType(enum.Enum):
    """ Enum for specifying what the type of a column is. """
    Blob = 0
    Video = 1

    @staticmethod
    def to_proto(protobufs, ty):
        if ty == ColumnType.Blob:
            return protobufs.Other
        elif ty == ColumnType.Video:
            return protobufs.Video
        else:
            raise ScannerException('Invalid column type')


class Job:
    def __init__(self, columns, name=None):
        self._columns = columns
        self._name = name

    def name(self):
        return self._name

    def op(self, db):
        return db.ops.Output(inputs=self._columns)
