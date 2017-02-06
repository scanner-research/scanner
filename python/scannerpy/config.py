import os
import toml
import sys
from common import *


class Config(object):
    def __init__(self, config_path=None):
        log.basicConfig(
            level=log.DEBUG,
            format='%(levelname)7s %(asctime)s %(filename)s:%(lineno)03d] %(message)s')
        self.config_path = config_path or self.default_config_path()
        config = self.load_config(self.config_path)
        try:
            self.scanner_path = config['scanner_path']

            if not os.path.isdir(self.scanner_path) or \
               not os.path.isdir(self.scanner_path + '/build') or \
               not os.path.isdir(self.scanner_path + '/scanner'):
                raise ScannerException("""Invalid Scanner directory. Make sure \
scanner_path in {} is correct and that Scanner is built correctly."""
                                       .format(self.scanner_path))

            sys.path.append('{}/build'.format(self.scanner_path))
            sys.path.append('{}/thirdparty/build/bin/storehouse/lib'
                            .format(self.scanner_path))

            from storehousepy import StorageConfig, StorageBackend
            storage = config['storage']
            storage_type = storage['type']
            self.db_path = str(storage['db_path'])
            if storage_type == 'posix':
                storage_config = StorageConfig.make_posix_config()
            elif storage_type == 'gcs':
                with open(storage['key_path']) as f:
                    key = f.read()
                storage_config = StorageConfig.make_gcs_config(
                    storage['cert_path'].encode('latin-1'),
                    key,
                    storage['bucket'].encode('latin-1'))
            else:
                raise ScannerException('Unsupported storage type {}'.format(storage_type))

            from scanner.metadata_pb2 import MemoryPoolConfig
            cfg = MemoryPoolConfig()
            if 'memory_pool' in config:
                memory_pool = config['memory_pool']
                if 'cpu' in memory_pool:
                    cfg.cpu.use_pool = memory_pool['cpu']['use_pool']
                    if cfg.cpu.use_pool:
                        cfg.cpu.free_space = self._parse_size_string(
                            memory_pool['cpu']['free_space'])
                if 'gpu' in memory_pool:
                    cfg.gpu.use_pool = memory_pool['gpu']['use_pool']
                    if cfg.gpu.use_pool:
                        cfg.gpu.free_space = self._parse_size_string(
                            memory_pool['gpu']['free_space'])
            self.memory_pool_config = cfg

            self.master_address = 'localhost:5001'
            if 'network' in config:
                network = config['network']
                if 'master_address' in network:
                    self.master_address = network['master_address']

            self.kernel_instances_per_node = 1
            self.work_item_size = 250
            self.io_item_size = 1000
            if 'job' in config:
                job = config['job']
                if 'kernel_instances_per_node' in job:
                    self.kernel_instances_per_node = job['kernel_instances_per_node']

        except KeyError as key:
            raise ScannerException('Scanner config missing key: {}'.format(key))
        self.storage_config = storage_config
        self.storage = StorageBackend.make_from_config(storage_config)

    def _parse_size_string(self, s):
        (prefix, suffix) = (s[:-1], s[-1])
        mults = {
            'G': 1024*1024*1024,
            'M': 1024*1024,
            'K': 1024
        }
        if suffix not in mults:
            raise ScannerException('Invalid size suffix in "{}"'.format(s))
        return int(prefix) * mults[suffix]

    @staticmethod
    def default_config_path():
        return '{}/.scanner.toml'.format(os.path.expanduser('~'))

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return toml.loads(f.read())
        except IOError:
            raise ScannerException('You need to setup your Scanner config. '
                                   'Run `python scripts/setup.py`.')
