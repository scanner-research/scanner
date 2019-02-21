import numpy as np
from collections import defaultdict
from enum import Enum


class ScannerException(Exception):
    pass


class DeviceType(Enum):
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


class DeviceHandle(object):
    def __init__(self, device, device_id):
        self.device = device
        self.device_id = device_id


# Class purely for type annotaiton
class FrameType(object):
    pass


BlobType = bytes


class ColumnType(Enum):
    """ Enum for specifying what the type of a column is. """
    Blob = 0
    Video = 1

    @staticmethod
    def to_proto(protobufs, ty):
        if ty == ColumnType.Blob:
            return protobufs.Bytes
        elif ty == ColumnType.Video:
            return protobufs.Video
        else:
            raise ScannerException('Invalid column type')


class CacheMode(Enum):
    Error = 1
    Ignore = 2
    Overwrite = 3
