from scannerpy.common import ScannerException, DeviceType, DeviceHandle, ColumnType, PerfParams
from scannerpy.types import FrameType
from scannerpy.job import Job
from scannerpy.client import Client, start_master, start_worker, CacheMode
from scannerpy.config import Config
from scannerpy.kernel import Kernel, KernelConfig
from scannerpy.op import register_python_op, SliceList
from scannerpy.protobufs import protobufs
from scannerpy.storage import NullElement, NamedVideoStream, NamedStream

# Check that grpc version is correct
def _check_grpc():
    import platform
    import subprocess
    import json
    import scannerpy.build_flags as bf

    def compatible_version(v1, v2):
        maj1, min1, bug1 = v1.split('.')
        maj2, min2, bug2 = v2.split('.')
        return maj1 == maj2 and min1 == min2

    os_type = platform.system()
    built_grpc_version = bf.get_grpc_version()
    if os_type == 'Darwin':
        has_brew = (subprocess.run('brew --help', shell=True,
                                   stdout=subprocess.DEVNULL).returncode == 0)
        has_grpc_brew = (subprocess.run('brew --help', shell=True,
                                        stdout=subprocess.DEVNULL).returncode == 0)
        has_grpc_python = (subprocess.run('pip3 show grpcio', shell=True,
                                          stdout=subprocess.DEVNULL).returncode == 0)
        if has_brew and has_grpc_brew and has_grpc_python:
            data = json.loads(subprocess.check_output('brew info --json grpc', shell=True))
            brew_version = data[0]['linked_keg']
            output = subprocess.check_output('pip3 list | grep grpcio', shell=True)
            python_version = [x for x in output.decode('utf-8').split(' ') if len(x) > 0][1]
            if not compatible_version(built_grpc_version, brew_version):
                print(('Warning: Scanner was built with GRPC version {:s}, '
                       'but the version installed via brew is {:s}. '
                       'Please reinstall Scanner to fix this issue.').format(
                           built_grpc_version, brew_version))
            if not compatible_version(built_grpc_version, python_version):
                print(('Warning: Scanner was built with GRPC version {:s}, '
                       'but the version installed via python is {:s}. '
                       'Please install grpcio == {:s} to fix this issue.').format(
                           built_grpc_version, python_version,
                           '.'.join(built_grpc_version.split('.')[0:2])))

_check_grpc()
