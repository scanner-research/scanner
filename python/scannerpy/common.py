import numpy as np
from collections import defaultdict
from enum import Enum
from multiprocessing import cpu_count
from psutil import virtual_memory


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


class PerfParams(object):
    """
        work_packet_size
          The size of the packets of intermediate elements to pass between
          operations. This parameter only affects performance and should not
          affect the output.

        io_packet_size
          The size of the packets of elements to read and write from Sources and
          sinks. This parameter only affects performance and should not
          affect the output. When reading and writing to high latency storage
          (such as the cloud), it is helpful to increase this value.
    """

    def __init__(self, work_packet_size, io_packet_size):
        self.work_packet_size = work_packet_size
        self.io_packet_size = io_packet_size

    @classmethod
    def manual(cls, work_packet_size, io_packet_size):
        def resolve(*args, **kwargs):
            return cls(work_packet_size, io_packet_size)
        return resolve

    @classmethod
    def estimate(cls, max_memory_util=0.8):
        def resolve(inputs, ops, tasks_in_queue_per_pu):
            max_size = 0
            for ins in inputs:
                try:
                    ins[0].estimate_size()
                except NotImplemented:
                    continue

                max_size = max(max([i.estimate_size() for i in ins]), max_size)

            has_gpu = False
            for op in ops:
                if hasattr(op, '_device') and op._device == DeviceType.GPU:
                    has_gpu = True

            if has_gpu:
                # TODO
                raise NotImplemented

            else:
                pipeline_instances = cpu_count()
                total_memory = virtual_memory().total * max_memory_util
                work_packet_size = int(total_memory / (tasks_in_queue_per_pu * max_size * pipeline_instances))
                io_packet_size = work_packet_size

            return cls(work_packet_size, io_packet_size)

        return resolve
