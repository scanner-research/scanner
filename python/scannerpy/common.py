import numpy as np
from collections import defaultdict
from enum import Enum
from multiprocessing import cpu_count
from psutil import virtual_memory
import GPUtil
import logging
import datetime
import math

log = logging.getLogger('scanner')
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            level = record.levelname[0]
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')[2:]
            if len(record.args) > 0:
                record.msg = '({})'.format(', '.join(
                    [str(x) for x in [record.msg] + list(record.args)]))
                record.args = ()
            return '{level} {time} {filename}:{lineno:03d}] {msg}'.format(
                level=level, time=time, **record.__dict__)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    log.addHandler(handler)


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
        r"""Explicitly provide values for each performance parameter.

        See class definition for explanation of each parameter.

        Parameters
        ----------
        work_packet_size

        io_packet_size
        """

        def resolve(*args, **kwargs):
            return cls(work_packet_size, io_packet_size)
        return resolve

    @classmethod
    def estimate(cls, max_memory_util=0.7, total_memory=None, work_io_ratio=0.2):
        r"""Guess the best value of each performance parameters given the computation graph.

        Parameters
        ----------
        max_memory_util
          Target maximum memory utilization as a fraction of the total system memory, e.g. 0.5 means Scanner
          should try to use 50% of the machine's memory.

        total_memory
          Total memory on the worker machines in bytes. Memory of the current machine will be used if none is
          is provided.

        work_io_ratio
          Ratio of work_packet_size to io_packet_size.
        """

        def resolve(inputs, ops, tasks_in_queue_per_pu):
            max_size = 0
            for ins in inputs:
                try:
                    ins[0].estimate_size()
                except NotImplementedError:
                    continue

                max_size = max(max([i.estimate_size() for i in ins]), max_size)

            if max_size == 0:
                log.warning('PerfParams.estimate could not estimate size of input stream elements, '
                            'falling back to conservative guess')
                return cls(10, 100)

            has_gpu = False
            for op in ops:
                if hasattr(op, '_device') and op._device == DeviceType.GPU:
                    has_gpu = True

            if has_gpu:
                gpus = GPUtil.getGPUs()
                pipeline_instances = len(gpus)
                max_memory = min([g.memoryTotal for g in gpus])
            else:
                pipeline_instances = cpu_count()
                max_memory = virtual_memory().total if total_memory is None else total_memory

            def fmt_bytes(n):
                exp = math.log2(n)
                if exp < 10:
                    return '{}B'.format(n)
                elif exp < 20:
                    return '{:.1f}KB'.format(n / (2**10))
                elif exp < 30:
                    return '{:.1f}MB'.format(n / (2**20))
                elif exp < 40:
                    return '{:.1f}GB'.format(n / (2**30))

            max_memory *= max_memory_util

            log.debug(
                """PERF PARAMS STATISTICS
                Maximum element size: {}
                Memory size: {}
                Pipeline instances: {}
                Tasks in queue per PU: {}
                """.format(fmt_bytes(max_size), fmt_bytes(max_memory), pipeline_instances, tasks_in_queue_per_pu))

            io_packet_size = int(max_memory / (tasks_in_queue_per_pu * max_size * pipeline_instances))
            io_packet_size = max(io_packet_size, 100)
            work_packet_size = int(io_packet_size * work_io_ratio)

            log.info('Estimated params: work packet size {}, io packet size {}'.format(
                work_packet_size, io_packet_size))

            return cls(work_packet_size, io_packet_size)

        return resolve
