import os
import toml
import sys
from subprocess import check_output
from common import *
from storehousepy import StorageConfig, StorageBackend


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

            config = self.default_config()
            path = self.default_config_path()
            with open(path, 'w') as f:
                f.write(toml.dumps(config))
            print('Wrote Scanner configuration to {}'.format(path))

        config = self.load_config(self.config_path)
        try:
            self.module_dir = os.path.dirname(os.path.realpath(__file__))
            build_path = self.module_dir + '/build'
            sys.path.append(build_path)

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

            self.master_address_base = 'localhost'
            self.master_port = '5001'
            self.worker_port = '5002'
            if 'network' in config:
                network = config['network']
                if 'master' in network:
                    self.master_address_base = network['master'].encode('ascii','ignore')
                if 'master_port' in network:
                    self.master_port = int(network['master_port'])
                if 'worker_port' in network:
                    self.worker_port = int(network['worker_port'])

            self.master_address = self.master_address_base + ':' + str(self.master_port)

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

    @staticmethod
    def default_config():
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
                'master': hostname,
                'master_port': '5001',
                'worker_port': '5002'
            }
        }
