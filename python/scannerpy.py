import toml
import os
import os.path
import sys
from enum import Enum
import grpc
from random import choice
from string import ascii_uppercase
import logging

class DeviceType(Enum):
    CPU = 0
    GPU = 1

class Config(object):
    """ TODO(wcrichto): document me """

    def __init__(self, config_path=None):
        self.config_path = config_path or self.default_config_path()
        config = self.load_config(self.config_path)
        try:
            self.scanner_path = config['scanner_path']
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
                logging.critical('Unsupported storage type {}'.format(storage_type))
                exit()

            from scanner.metadata_pb2 import MemoryPoolConfig
            self.memory_pool_config = MemoryPoolConfig()
            if 'memory_pool' in config:
                memory_pool = config['memory_pool']
                self.memory_pool_config.use_pool = memory_pool['use_pool']
                if self.memory_pool_config.use_pool:
                    self.memory_pool_config.pool_size = memory_pool['pool_size']
            else:
                self.memory_pool_config.use_pool = False

            if 'network' in config:
                network = config['network']
                if 'master_address' in network:
                    self.master_address = network['master_address']
                else:
                    self.master_address = 'localhost:5001'
            else:
                self.master_address = 'localhost:5001'

        except KeyError as key:
            logging.critical('Scanner config missing key: {}'.format(key))
            exit()
        self.storage_config = storage_config
        self.storage = StorageBackend.make_from_config(storage_config)

    @staticmethod
    def default_config_path():
        return '{}/.scanner.toml'.format(os.path.expanduser('~'))

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return toml.loads(f.read())
        except IOError:
            logging.critical('Error: you need to setup your Scanner config. Run `python scripts/setup.py`.')
            exit()


class Collection:
    def __init__(self, descriptor):
        self._descriptor = descriptor

    def tables(self):
        return list(self._descriptor.tables)


class Database:
    def __init__(self, config_path=None):
        self.config = Config(config_path)

        # Load all protobuf types
        import scanner.metadata_pb2 as metadata_types
        import scanner.engine.rpc_pb2 as rpc_types
        import scanner.kernels.args_pb2 as arg_types
        import scanner_bindings as bindings
        self._metadata_types = metadata_types
        self._rpc_types = rpc_types
        self._arg_types = arg_types
        self._bindings = bindings

        # Setup database metadata
        self._db_path = self.config.db_path
        self._pydb_path = '{}/pydb'.format(self._db_path)
        self._storage = self.config.storage
        self._master_address = self.config.master_address
        self._db_params = self._bindings.make_database_parameters(
            self.config.storage_config,
            self.config.memory_pool_config.SerializeToString(),
            self._db_path)
        self.evaluators = EvaluatorGenerator(self)

        # Initialize database if it does not exist
        if not os.path.isfile('{}/db_metadata.bin'.format(self._db_path)):
            self._bindings.create_database(self.config.storage_config, self._db_path)
            try:
                os.mkdir(self._pydb_path)
            except OSError:
                logging.critical('Tried to use Scanner database not created in Python')
                exit()
            collections = self._metadata_types.CollectionsDescriptor()
            with open('{}/descriptor.bin'.format(self._pydb_path), 'w') as f:
                f.write(collections.SerializeToString())

        # Load collection metadata from disk
        self._collections = self._metadata_types.CollectionsDescriptor()
        with open('{}/descriptor.bin'.format(self._pydb_path)) as f:
            self._collections.ParseFromString(f.read())

        # Initialize gRPC channel with master server
        channel = grpc.insecure_channel(self._master_address)
        self._master = self._rpc_types.MasterStub(channel)

    def start_master(self):
        return self._bindings.start_master(self._db_params)

    def start_worker(self, master_address=None):
        return self._bindings.start_worker(self._db_params, self._master_address)

    def new_collection(self, collection_name, table_names, videos):
        if collection_name in self._collections.names:
            logging.critical('Collection with name {} already exists' \
                             .format(collection_name))
            exit()

        last_id = self._collections.ids[-1] if len(self._collections.ids) > 0 else -1
        new_id = last_id + 1
        self._collections.ids.append(new_id)
        self._collections.names.append(collection_name)
        collection = self._metadata_types.CollectionDescriptor()
        collection.tables.extend(table_names)

        with open('{}/collection_{}.bin'.format(self._pydb_path, new_id), 'w') as f:
            f.write(collection.SerializeToString())

        return Collection(collection)

    def get_collection(self, name):
        index = self._collections.names[:].index(name)
        id = self._collections.ids[index]
        collection = self._metadata_types.Collection()
        with open('{}/collection_{}.bin'.format(self._pydb_path, index)) as f:
            collection.ParseFromString(f.read())
        return Collection(collection)

    def ingest_video(self, table_name, video):
        self._bindings.ingest_videos(
            self.config.storage_config,
            self._db_path,
            [table_name],
            [video])

    def ingest_video_collection(self, collection_name, videos):
        table_names = ['{}_{:03d}'.format(collection_name, i)
                       for i in range(len(videos))]
        self._bindings.ingest_videos(
            self.config.storage_config,
            self._db_path,
            table_names,
            videos)
        return self.new_collection(collection_name, table_names, videos)

    def make_sampler(self):
        return Sampler(self)

    def _process_dag(self, evaluator):
        if isinstance(evaluator, list):
            for i in range(len(evaluator) - 1):
                out_cols = self._bindings.get_output_columns(evaluator[i]._name)
                evaluator[i+1]._inputs = [(evaluator[i], out_cols)]
            evaluator = evaluator[-1]
        if evaluator._name != "OutputTable":
            out_cols = self._bindings.get_output_columns(evaluator._name)
            print out_cols
            evaluator = Evaluator.output([(evaluator, out_cols)])
        return []

    def run(self, tasks, evaluator, output_collection=None, job_name=None):
        if isinstance(tasks, Collection):
            sampler = self.make_sampler()
            tasks = sampler.all_frames(tasks.tables())
        job_params = self._rpc_types.JobParameters()
        job_name = job_name or ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.task_set.tasks.extend(tasks)
        job_params.task_set.evaluators.extend(self._process_dag(evaluator))
        exit()
        self._master.NewJob(job_params)


class Sampler:
    def __init__(self, db):
        self._db = db

    def all_frames(self, videos):
        tasks = []
        for table in videos:
            task = self._db._metadata_types.Task()
            task.output_table_name = "TODO"
            row_count = 100 # TODO: extract this
            column_names = ["frame", "frame_info"] # TODO: extract this
            sample = task.samples.add()
            sample.table_name = table
            sample.column_names.extend(column_names)
            sample.rows.extend(range(row_count))
            tasks.append(task)
        return tasks


class EvaluatorGenerator:
    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        if not self._db._bindings.has_evaluator(name):
            logging.critical('Evaluator {} does not exist'.format(name))
            exit()
        def make_evaluator(**kwargs):
            inputs = kwargs.pop('inputs', [])
            device = kwargs.pop('device', None)
            if device is None:
                logging.critical('Must specify device type')
                exit()
            return Evaluator(name, inputs, device, kwargs)
        return make_evaluator


class Evaluator:
    def __init__(self, name, inputs, device, args):
        self._name = name
        self._inputs = inputs
        self._device = device
        self._args = args

    @classmethod
    def input(cls):
        return cls("InputTable", [])

    @classmethod
    def output(cls, inputs):
        return cls("OutputTable", inputs)
