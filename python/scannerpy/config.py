import os
import toml
import sys
from subprocess import check_output
from common import *


def read_line(s):
    return sys.stdin.readline().strip()

class Config(object):
    def __init__(self, config_path=None):
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

            config = self._default_config()
            path = self.default_config_path()
            with open(path, 'w') as f:
                f.write(toml.dumps(config))
            print('Wrote Scanner configuration to {}'.format(path))

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

            self.master_address = 'localhost:5001'
            if 'network' in config:
                network = config['network']
                if 'master_address' in network:
                    self.master_address = network['master_address']

        except KeyError as key:
            raise ScannerException('Scanner config missing key: {}'.format(key))
        self.storage_config = storage_config
        self.storage = StorageBackend.make_from_config(storage_config)

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

    def _default_config(self):
        hostname = check_output(['hostname', '-A']).split(' ')[0]

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
                'master': hostname + ':5001'
            }
        }
