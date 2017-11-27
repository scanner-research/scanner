from __future__ import absolute_import, division, print_function, unicode_literals
import os
import toml
import sys
from subprocess import check_output

from scannerpy.common import *
from storehousepy import StorageConfig, StorageBackend


def read_line(s):
    return sys.stdin.readline().strip()


class Config(object):
    def __init__(self, config_path=None, db_path=None):
        log.basicConfig(
            level=log.DEBUG,
            format='%(levelname)s %(asctime)s %(filename)s:%(lineno)03d] %(message)s')
        self.config_path = config_path or self.default_config_path()

        if not os.path.isfile(self.config_path):
            sys.stdout.write(
                'Your Scanner configuration file does not exist. Create one? '
                '[Y/n] ')
            if sys.stdin.readline().strip().lower() == 'n':
                exit()

            config = self.default_config()
            path = self.default_config_path()
            with open(path, 'w') as f:
                f.write(toml.dumps(config))
            print('Wrote Scanner configuration to {}'.format(path))

        self.config = self.load_config(self.config_path)
        config = self.config
        try:
            self.module_dir = os.path.dirname(os.path.realpath(__file__))
            build_path = self.module_dir + '/build'
            sys.path.append(build_path)

            if db_path is not None:
                self.db_path = db_path
            else:
                storage = config['storage']
                self.db_path = str(storage['db_path'])
            storage_config = self._make_storage_config(config)

            self.master_address = 'localhost'
            self.master_port = '5001'
            self.worker_port = '5002'
            if 'network' in config:
                network = config['network']
                if 'master' in network:
                    self.master_address = network['master'].encode('ascii','ignore')
                if 'master_port' in network:
                    self.master_port = network['master_port'].encode('ascii', 'ignore')
                if 'worker_port' in network:
                    self.worker_port = network['worker_port'].encode('ascii', 'ignore')

        except KeyError as key:
            raise ScannerException('Scanner config missing key: {}'.format(key))
        self.storage_config = storage_config
        self.storage = StorageBackend.make_from_config(storage_config)

    def _make_storage_config(self, config):
        storage = config['storage']
        storage_type = storage['type']
        if storage_type == 'posix':
            storage_config = StorageConfig.make_posix_config()
        elif storage_type == 'gcs':
            storage_config = StorageConfig.make_gcs_config(
                storage['bucket'].encode('latin-1'))
        elif storage_type == 's3':
            storage_config = StorageConfig.make_s3_config(
                storage['bucket'].encode('latin-1'),
                storage['region'].encode('latin-1'),
                storage['endpoint'].encode('latin-1'))
        else:
            raise ScannerException(
                'Unsupported storage type {}'.format(storage_type))
        return storage_config

    @staticmethod
    def default_config_path():
        return os.path.expanduser('~') + '/.scanner.toml'

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return toml.loads(f.read())
        except IOError:
            raise ScannerException('Scanner config file does not exist: {}'
                                   .format(path))

    @staticmethod
    def default_config():
        hostname = check_output(['hostname']).strip()

        scanner_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..'))

        db_path = os.path.expanduser('~') + '/.scanner_db'

        return {
            'scanner_path': scanner_path,
            'storage': {
                'type': 'posix',
                'db_path': db_path,
            },
            'network': {
                'master': hostname,
                'master_port': '5001',
                'worker_port': '5002'
            }
        }

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        # Get rid of the storehouse objects
        state.pop('storage_config', None)
        state.pop('storage', None)
        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, newstate):
        self.module_dir = os.path.dirname(os.path.realpath(__file__))
        build_path = self.module_dir + '/build'
        if not build_path in sys.path:
            sys.path.append(build_path)
        sys.stdout.flush()

        sc = self._make_storage_config(newstate['config'])
        newstate['storage_config'] = sc
        newstate['storage'] = StorageBackend.make_from_config(sc)
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(newstate)
