import os
import toml
import sys
from subprocess import check_output
import errno

from scannerpy.common import *
from storehouse import StorageConfig, StorageBackend


def read_line(s):
    return sys.stdin.readline().strip()


# https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Config(object):
    def __init__(self, config_path=None, db_path=None):
        self.config_path = config_path or self.default_config_path()

        # Prompt to create Scanner configuration file if it doesn't already exit
        if not os.path.isfile(self.config_path):
            self._create_config_prompt()

        # Load configuration from file
        config = self._load_config(self.config_path)
        self.config = config

        # Extract information from configuration
        try:
            # Add build directory to Python import path
            self.module_dir = os.path.dirname(os.path.realpath(__file__))
            build_path = os.path.join(self.module_dir, 'build')
            sys.path.append(build_path)

            # Determine path to database
            if db_path is not None:
                self.db_path = db_path
            else:
                storage = config['storage']
                self.db_path = str(storage['db_path'])
            mkdir_p(self.db_path)

            # Create connector to Storehouse
            storage_config = self._make_storage_config(config)
            self.storage_config = storage_config
            self.storage = StorageBackend.make_from_config(storage_config)

            # Configure network settings
            self.master_address = 'localhost'
            self.master_port = '5001'
            self.worker_port = '5002'
            if 'network' in config:
                network = config['network']
                if 'master' in network:
                    self.master_address = network['master']
                if 'master_port' in network:
                    self.master_port = network['master_port']
                if 'worker_port' in network:
                    self.worker_port = network['worker_port']

        except KeyError as key:
            raise ScannerException(
                'Scanner config missing key: {}'.format(key))

    def _make_storage_config(self, config):
        storage = config['storage']
        storage_type = storage['type']
        if storage_type == 'posix':
            storage_config = StorageConfig.make_posix_config()
        elif storage_type == 'gcs':
            storage_config = StorageConfig.make_gcs_config(storage['bucket'])
        elif storage_type == 's3':
            storage_config = StorageConfig.make_s3_config(
                storage['bucket'], storage['region'], storage['endpoint'])
        else:
            raise ScannerException(
                'Unsupported storage type {}'.format(storage_type))
        return storage_config

    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return toml.loads(f.read())
        except IOError:
            raise ScannerException('Scanner config file does not exist: {}'
                                   .format(path))

    def _create_config_prompt(self):
        sys.stdout.write(
            'Your Scanner configuration file ({}) does not exist. Create one? '
            '[Y/n] '.format(self.config_path))
        sys.stdout.flush()
        if sys.stdin.readline().strip().lower() == 'n':
            print(
                'Exiting script. Please create a Scanner configuration file or re-run this script and follow '
                'the dialogue.')
            exit()

        config = self.default_config()
        path = self.default_config_path()
        mkdir_p(os.path.split(path)[0])
        with open(path, 'w') as f:
            f.write(toml.dumps(config))
        print('Wrote Scanner configuration to {}'.format(path))

    @staticmethod
    def default_config_path():
        return os.path.expanduser('~/.scanner/config.toml')

    @staticmethod
    def default_config():
        hostname = 'localhost'

        db_path = os.path.expanduser('~/.scanner/db')

        return {
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
