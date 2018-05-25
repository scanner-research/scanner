# dlopen scanner library to avoid lookup
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if sys.platform == 'linux' or sys.platform == 'linux2':
    EXT = '.so'
else:
    EXT = '.dylib'

import ctypes
__sc = ctypes.cdll.LoadLibrary(os.path.join(SCRIPT_DIR, 'lib', 'libscanner' + EXT))

del EXT
del SCRIPT_DIR

from scannerpy.common import ScannerException, DeviceType, DeviceHandle, FrameType, ColumnType
from scannerpy.job import Job
from scannerpy.database import Database, ProtobufGenerator, start_master, start_worker
from scannerpy.config import Config
from scannerpy.kernel import Kernel, KernelConfig
from scannerpy.op import register_python_op
