"""
Python bindings for Scanner.
"""

import toml
import os
import os.path
import sys
import grpc
import struct
import cv2
import importlib
import socket
import math
import numpy as np
import logging as log
import json
from subprocess import Popen, PIPE
from enum import Enum
from random import choice
from string import ascii_uppercase
from collections import defaultdict


class DeviceType(Enum):
    """ Enum for specifying where an Evaluator should run. """
    CPU = 0
    GPU = 1

    @staticmethod
    def to_proto(db, device):
        if device == DeviceType.CPU:
            return db._metadata_types.CPU
        elif device == DeviceType.GPU:
            return db._metadata_types.GPU
        else:
            raise ScannerException('Invalid device type')


class Config(object):
    def __init__(self, config_path=None):
        log.basicConfig(
            level=log.DEBUG,
            format='%(levelname)7s %(asctime)s %(filename)s:%(lineno)03d] %(message)s')
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

            if 'network' in config:
                network = config['network']
                if 'master_address' in network:
                    self.master_address = network['master_address']
                else:
                    self.master_address = 'localhost:5001'
            else:
                self.master_address = 'localhost:5001'

            if 'job' in config:
                job = config['job']
                if 'kernel_instances_per_node' in job:
                    self.kernel_instances_per_node = job['kernel_instances_per_node']
                else:
                    self.kernel_instances_per_node = 1

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
        if not suffix in mults:
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
            raise ScannerException('You need to setup your Scanner config. Run `python scripts/setup.py`.')


class ScannerException(Exception): pass


class Database:
    """
    Entrypoint for all Scanner operations.

    Attributes:
        config: The Config object for the database.
        evaluators: An EvaluatorGenerator object for computation creation.
        protobufs: TODO(wcrichto)
    """

    def __init__(self, config_path=None):
        """
        Initializes a Scanner database.

        This will create a database at the `db_path` specified in the config
        if none exists.

        Kwargs:
            config_path: Path to a Scanner configuration TOML, by default
                         assumed to be `~/.scanner.toml`.

        Returns:
            A database instance.
        """
        self.config = Config(config_path)

        # Load all protobuf types
        import scanner.metadata_pb2 as metadata_types
        import scanner.engine.rpc_pb2 as rpc_types
        import scanner.kernels.args_pb2 as arg_types
        import scanner_bindings as bindings
        self._metadata_types = metadata_types
        self._rpc_types = rpc_types
        self._arg_types = [arg_types]
        self._bindings = bindings

        # Setup database metadata
        self._db_path = self.config.db_path
        self._storage = self.config.storage
        self._master_address = self.config.master_address
        self._db_params = self._bindings.make_database_parameters(
            self.config.storage_config,
            self.config.memory_pool_config.SerializeToString(),
            self._db_path)

        self.evaluators = EvaluatorGenerator(self)
        self.protobufs = ProtobufGenerator(self)

        # Initialize database if it does not exist
        pydb_path = '{}/pydb'.format(self._db_path)
        if not os.path.isfile('{}/db_metadata.bin'.format(self._db_path)):
            self._bindings.create_database(self.config.storage_config, self._db_path)
            os.mkdir(pydb_path)
            self._collections = self._metadata_types.CollectionsDescriptor()
            self._update_collections()

        if not os.path.isdir(pydb_path):
            raise ScannerException(
                'Scanner database at {} was not made via Python' \
                .format(self._db_path))

        # Load database descriptors from disk
        self._collections = self._load_descriptor(
            self._metadata_types.CollectionsDescriptor,
            'pydb/descriptor.bin')

        # Initialize gRPC channel with master server
        channel = grpc.insecure_channel(self._master_address)
        self._master = self._rpc_types.MasterStub(channel)

    def get_build_flags(self):
        """
        Gets the g++ build flags for compiling custom evaluators.

        For example, to compile a custom kernel:
        \code{.sh}
        export SCANNER_FLAGS=`python -c "import scannerpy as sp; print(sp.Database().get_build_flags())"`
        g++ mykernel.cpp -o mylib.so `echo $SCANNER_FLAGS`
        \endcode

        Returns:
           A flag string.
        """

        include_dirs = self._bindings.get_include().split(";")
        flags = '{include} -std=c++11 -fPIC -shared -L{libdir} -lscanner {other}'
        return flags.format(
            include=" ".join(["-I " + d for d in include_dirs]),
            libdir='{}/build'.format(self.config.scanner_path),
            other=self._bindings.other_flags())

    def _load_descriptor(self, descriptor, path):
        d = descriptor()
        d.ParseFromString(self._storage.read('{}/{}'.format(self._db_path, path)))
        return d

    def _save_descriptor(self, descriptor, path):
        self._storage.write(
            '{}/{}'.format(self._db_path, path),
            descriptor.SerializeToString())

    def _load_db_metadata(self):
        return self._load_descriptor(
            self._metadata_types.DatabaseDescriptor,
            'db_metadata.bin')

    def start_master(self, block=False):
        """
        Starts a master server on the current node.

        Scanner clusters require one master server to coordinate computation.
        If the returned value falls out of scope and is garbage collected,
        the server will exit, so make sure to bind the result to a variable.

        Kwargs:
            block: Whether to block on the master creation call.

        Returns:
            An opaque handle to the master.
        """

        return self._bindings.start_master(self._db_params, block)

    def start_worker(self, master_address=None, block=False):
        """
        Starts a worker on the current node.

        Each node can have one or many workers (multiple workers can be used
        to run multiple kernels per node that require process isolation). If the
        returned value falls out of scope and is garbage collected, the server
        will exit, so make sure to bind the result to a variable. The master must
        be started before the worker is created.

        Kwargs:
            master_address: Address and port of the master node.
            block: Whether to block on the worker creation call.

        Returns:
            An opaque handle to the worker.
        """

        worker_params = self._bindings.default_worker_params()
        if master_address is None:
            master_address = self._master_address
        return self._bindings.start_worker(self._db_params, worker_params,
                                           master_address, block)

    def _run_remote_cmd(self, host, cmd):
        local_ip = socket.gethostbyname(socket.gethostname())
        if socket.gethostbyname(host) == local_ip:
            return Popen(cmd, shell=True)
        else:
            return Popen("ssh {} {}".format(host, cmd), shell=True)

    def start_cluster(self, master, workers):
        """
        Convenience method for starting a Scanner cluster.

        This should be run as a background/tmux/etc. script.

        Args:
            master: ssh-able address of the master node.
            workers: list of ssh-able addresses of the worker nodes.
        """
        master_cmd = 'python -c "from scannerpy import Database as Db; Db().start_master(True)"'
        worker_cmd = 'python -c "from scannerpy import Database as Db; Db().start_worker(\'{}:5001\', True)"' \
                     .format(master)

        master = self._run_remote_cmd(master, master_cmd)
        workers = [self._run_remote_cmd(w, worker_cmd) for w in workers]
        master.wait()
        for worker in workers:
            worker.wait()

    def load_evaluator(self, so_path, proto_path=None):
        """
        Loads a custom evaluator into the Scanner runtime.

        By convention, if the evaluator requires arguments from Python, it must
        have a protobuf message called <EvaluatorName>Args, e.g. BlurArgs or
        HistogramArgs, and the path to that protobuf should be provided.

        Args:
            so_path: Path to the custom evaluator's shared object file.

        Kwargs:
            proto_path: Path to the custom evaluator's arguments protobuf
                        if one exists.
        """
        if proto_path is not None:
            (proto_dir, mod_file) = os.path.split(proto_path)
            sys.path.append(proto_dir)
            (mod_name, _) = os.path.splitext(mod_name)
            self._arg_types.append(importlib.import_module(mod_name))
        self._bindings.load_evaluator(so_path)

    def _update_collections(self):
        self._save_descriptor(self._collections, 'pydb/descriptor.bin')

    def delete_collection(self, collection_name):
        if not collection_name in self._collections.names:
            raise ScannerException('Collection with name {} does not exist' \
                                   .format(collection_name))

        index = self._collections.names[:].index(collection_name)
        id = self._collections.ids[index]
        del self._collections.names[index]
        del self._collections.ids[index]

        os.remove('{}/pydb/collection_{}.bin'.format(self._db_path, id))

    def new_collection(self, collection_name, table_names, force=False):
        """
        Creates a new Collection from a list of tables.

        Args:
            collection_name: String name of the collection to create.
            table_names: List of table name strings to put in the collection.

        Kwargs:
            force: TODO(wcrichto)

        Returns:
            The new Collection object.
        """

        if collection_name in self._collections.names:
            if force:
                self.delete_collection(collection_name)
            else:
                raise ScannerException(
                    'Collection with name {} already exists' \
                    .format(collection_name))

        last_id = self._collections.ids[-1] if len(self._collections.ids) > 0 else -1
        new_id = last_id + 1
        self._collections.ids.append(new_id)
        self._collections.names.append(collection_name)
        self._update_collections()
        collection = self._metadata_types.CollectionDescriptor()
        collection.tables.extend(table_names)
        self._save_descriptor(collection, 'pydb/collection_{}.bin'.format(new_id))

        return self.collection(collection_name)


    def ingest_video(self, table_name, video):
        """
        Creates a Table from a video.

        Args:
            table_name: String name of the Table to create.
            video: Path to the video.

        Returns:
            The newly created Table object.
        """

        self._bindings.ingest_videos(
            self.config.storage_config,
            self._db_path,
            [table_name],
            [video])
        return self.table(table_name)

    def ingest_video_collection(self, collection_name, videos):
        """
        Creates a Collection from a list of videos.

        Args:
            collection_name: String name of the Collection to create.
            videos: List of video paths.

        Returns:
            The newly created Collection object.
        """
        table_names = ['{}:{:03d}'.format(collection_name, i)
                       for i in range(len(videos))]
        collection = self.new_collection(collection_name, table_names)
        self._bindings.ingest_videos(
            self.config.storage_config,
            self._db_path,
            table_names,
            videos)
        return collection

    def has_collection(self, name):
        return name in self._collections.names

    def collection(self, name):
        index = self._collections.names[:].index(name)
        id = self._collections.ids[index]
        collection = self._load_descriptor(
            self._metadata_types.CollectionDescriptor,
            'pydb/collection_{}.bin'.format(index))
        return Collection(self, name, collection)

    def table(self, name):
        db_meta = self._load_db_metadata()

        if isinstance(name, basestring):
            table_id = None
            for table in db_meta.tables:
                if table.name == name:
                    table_id = table.id
                    break
            if table_id is None:
                raise ScannerException('Table with name {} not found'.format(name))
        elif isinstance(name, int):
            table_id = name
        else:
            raise ScannerException('Invalid table identifier')

        descriptor = self._load_descriptor(
            self._metadata_types.TableDescriptor,
            'tables/{}/descriptor.bin'.format(table_id))
        return Table(self, descriptor)

    def sampler(self):
        return Sampler(self)

    def profiler(self, job_name):
        db_meta = self._load_db_metadata()
        if isinstance(job_name, basestring):
            job_id = None
            for job in db_meta.jobs:
                if job.name == job_name:
                    job_id = job.id
                    break
            if job_id is None:
                raise ScannerException('Job name {} does not exist'.format(job_name))
        else:
            job_id = job_name

        return Profiler(self, job_id)

    def _toposort(self, evaluator):
        edges = defaultdict(list)
        in_edges_left = defaultdict(int)
        start_node = None

        explored_nodes = set()
        stack = [evaluator]
        while len(stack) > 0:
            c = stack.pop()
            explored_nodes.add(c)
            if (c._name == "InputTable"):
                start_node = c
                continue
            elif len(c._inputs) == 0:
                input = Evaluator.input(self)
                # TODO(wcrichto): allow non-frame input
                c._inputs = [(input, ["frame", "frame_info"])]
                start_node = input
            for (parent, _) in c._inputs:
                edges[parent].append(c)
                in_edges_left[c] += 1

                if parent not in explored_nodes:
                    stack.append(parent)

        eval_sorted = []
        eval_index = {}
        stack = [start_node]
        while len(stack) > 0:
            c = stack.pop()
            eval_sorted.append(c)
            eval_index[c] = len(eval_sorted) - 1
            for child in edges[c]:
                in_edges_left[child] -= 1
                if in_edges_left[child] == 0:
                    stack.append(child)

        return [e.to_proto(eval_index) for e in eval_sorted]

    def _process_dag(self, evaluator):
        # If evaluators are passed as a list (e.g. [transform, caffe])
        # then hook up inputs to outputs of adjacent evaluators
        if isinstance(evaluator, list):
            for i in range(len(evaluator) - 1):
                if len(evaluator[i+1]._inputs) > 0: continue
                if evaluator[i]._name == "InputTable":
                    out_cols = ["frame", "frame_info"]
                else:
                    out_cols = self._bindings.get_output_columns(evaluator[i]._name)
                evaluator[i+1]._inputs = [(evaluator[i], out_cols)]
            evaluator = evaluator[-1]

        # If the user doesn't explicitly specify an OutputTable, assume that
        # it's all the output columns of the last evaluator.
        if evaluator._name != "OutputTable":
            out_cols = self._bindings.get_output_columns(str(evaluator._name))
            evaluator = Evaluator.output(self, [(evaluator, out_cols)])

        return self._toposort(evaluator)

    def run(self, tasks, evaluator, output_collection=None, job_name=None, force=False):
        """
        Runs a computation over a set of inputs.

        Args:
            tasks: The set of inputs to run the computation on. If tasks is a
                   Collection, then the computation is run on all frames of all
                   tables in the collection. Otherwise, tasks should be generated
                   by the Sampler.
            evaluator: The computation to run. Evaluator is either a list of
                   evaluators to run in sequence, or a DAG with the output node
                   passed in as the argument.

        Kwargs:
            output_collection: If this is not None, then a new collection with
                               this name will be created for all the output
                               tables.
            job_name: An optional name to assign the job. It will be randomly
                      generated if none is given.
            force: TODO(wcrichto)

        Returns:
            Either the output Collection is output_collection is specified
            or a list of Table objects.
        """
        # Ping master and start master/worker locally if they don't exist.
        try:
            self._master.Ping(self._rpc_types.Empty())
        except grpc.RpcError as e:
            status = e.code()
            if status == grpc.StatusCode.UNAVAILABLE:
                log.warn("Master not started, creating temporary master/worker")
                # If they get GC'd then the masters/workers will die, so persist
                # them until the database object dies
                self._ignore1 = self.start_master()
                self._ignore2 = self.start_worker()
            elif status == grpc.StatusCode.OK:
                pass
            else:
                raise ScannerException('Master ping errored with status: {}' \
                                   .format(status))

        # If the input is a collection, assume user is running over all frames
        input_is_collection = isinstance(tasks, Collection)
        if input_is_collection:
            sampler = self.sampler()
            tasks = sampler.all(tasks)

        # If the output should be a collection, then set the table names
        if output_collection is not None:
            if self.has_collection(output_collection) and not force:
                raise ScannerException(
                    'Collection with name {} already exists' \
                    .format(output_collection))
            for task in tasks:
                new_name = '{}:{}'.format(
                    output_collection,
                    task.output_table_name.split(':')[-1])
                task.output_table_name = new_name

        job_params = self._rpc_types.JobParameters()
        # Generate a random job name if none given
        job_name = job_name or ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.task_set.tasks.extend(tasks)
        job_params.task_set.evaluators.extend(self._process_dag(evaluator))
        job_params.kernel_instances_per_node = self.config.kernel_instances_per_node

        # Execute job via RPC
        try:
            self._master.NewJob(job_params)
        except grpc.RpcError as e:
            raise ScannerException('Job failed with error: {}'.format(e))

        # Return a new collection if the input was a collection, otherwise
        # return a table list
        table_names = [task.output_table_name for task in tasks]
        if output_collection is not None:
            return self.new_collection(output_collection, table_names, force)
        else:
            return [self.table(t) for t in table_names]



class Sampler:
    """
    Utility for specifying which frames of a video (or which rows of a table)
    to run a computation over.
    """

    def __init__(self, db):
        self._db = db

    def _convert_collection(self, videos):
        if isinstance(videos, Collection):
            return [(t, '') for t in videos.table_names()]
        else:
            return videos

    def all(self, videos):
        return self.strided(videos, 1)

    def strided(self, videos, stride):
        videos = self._convert_collection(videos)
        tasks = []
        for video in videos:
            table = self._db.table(video[0])
            task = self.strided_range(video, 0, table.num_rows(), stride)
            tasks.append(task)
        return tasks

    def range(self, video, start, end):
        if isinstance(video, list) or isinstance(video, Collection):
            raise ScannerException('Sampler.range only takes a single video')

        return self.strided_range(video, start, end, 1)

    def strided_range(self, video, start, end, stride):
        if isinstance(video, list) or isinstance(video, Collection):
            raise ScannerException('Sampler.strided_range only takes a single video')
        if not isinstance(video, tuple):
            raise ScannerException("""Error: sampler input must either be a collection \
or (input_table, output_table) pair')""")

        (input_table_name, output_table_name) = video
        task = self._db._metadata_types.Task()
        task.output_table_name = output_table_name
        input_table = self._db.table(input_table_name)
        num_rows = input_table.num_rows()
        column_names = [c.name() for c in input_table.columns()]
        sample = task.samples.add()
        sample.table_name = input_table_name
        sample.column_names.extend(column_names)
        sample.rows.extend(
            range(min(start, num_rows), min(end, num_rows), stride))
        return task


class ProtobufGenerator:
    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        for mod in self._db._arg_types:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise ScannerException('No protobuf with name {}'.format(name))


class EvaluatorGenerator:
    """
    Creates Evaluator instances to define a computation.

    When a particular evaluator is requested from the generator, e.g.
    `db.evaluators.Histogram`, the generator does a dynamic lookup for the
    evaluator in a C++ registry.
    """

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        if name == 'Input':
            return lambda: Evaluator.input(self._db)
        elif name == 'Output':
            return lambda inputs: Evaluator.output(self._db, inputs)

        if not self._db._bindings.has_evaluator(name):
            raise ScannerException('Evaluator {} does not exist'.format(name))
        def make_evaluator(**kwargs):
            inputs = kwargs.pop('inputs', [])
            device = kwargs.pop('device', DeviceType.CPU)
            proto = kwargs.pop('proto', None)
            return Evaluator(self._db, name, inputs, device, proto, kwargs)
        return make_evaluator


class Evaluator:
    def __init__(self, db, name, inputs, device, proto, args):
        self._db = db
        self._name = name
        self._inputs = inputs
        self._device = device
        self._args = args
        self._proto = proto

    @classmethod
    def input(cls, db):
        # TODO(wcrichto): allow non-frame inputs
        return cls(db, "InputTable", [(None, ["frame", "frame_info"])],
                   DeviceType.CPU, None, {})

    @classmethod
    def output(cls, db, inputs):
        return cls(db, "OutputTable", inputs, DeviceType.CPU, None, {})

    def to_proto(self, indices):
        e = self._db._metadata_types.Evaluator()
        e.name = self._name

        for (in_eval, cols) in self._inputs:
            inp = e.inputs.add()
            idx = indices[in_eval] if in_eval is not None else -1
            inp.evaluator_index = idx
            inp.columns.extend(cols)

        e.device_type = DeviceType.to_proto(self._db, self._device)

        # To convert arguments, we search for a protobuf with the name
        # {Evaluator}Args (e.g. BlurArgs, HistogramArgs) in the args.proto
        # module, and fill that in with keys from the args dict.
        if len(self._args) > 0:
            proto_name = self._name + 'Args' \
                         if self._proto is None else self._proto + 'Args'
            args_proto = self._db.proto(proto_name)()
            for k, v in self._args.iteritems():
                try:
                    setattr(args_proto, k, v)
                except AttributeError:
                    # If the attribute is a nested proto, we can't assign directly,
                    # so copy from the value.
                    getattr(args_proto, k).CopyFrom(v)
                e.kernel_args = args_proto.SerializeToString()

        return e


class Collection:
    """
    A set of Table objects.
    """

    def __init__(self, db, name, descriptor):
        self._db = db
        self._name = name
        self._descriptor = descriptor

    def name(self):
        return self._name

    def table_names(self):
        return list(self._descriptor.tables)

    def tables(self, index=None):
        tables = [self._db.table(t) for t in self._descriptor.tables]
        return tables[index] if index is not None else tables


class Column:
    """
    A column of a Table.
    """

    def __init__(self, table, descriptor):
        self._table = table
        self._descriptor = descriptor
        self._db = table._db
        self._storage = table._db.config.storage
        self._db_path = table._db.config.db_path

    def name(self):
        return self._descriptor.name

    def _load_output_file(self, item_id, rows, fn=None):
        assert len(rows) > 0

        path = '{}/tables/{}/{}_{}.bin'.format(
            self._db_path, self._table._descriptor.id,
            self._descriptor.id, item_id)
        try:
            contents = self._storage.read(path)
        except UserWarning:
            raise ScannerException('Path {} does not exist'.format(path))

        lens = []
        start_pos = None
        pos = 0
        (num_rows,) = struct.unpack("l", contents[:8])

        i = 8
        rows = rows if len(rows) > 0 else range(num_rows)
        for fi in rows:
            (buf_len,) = struct.unpack("l", contents[i:i+8])
            i += 8
            old_pos = pos
            pos += buf_len
            if start_pos is None:
                start_pos = old_pos
            lens.append(buf_len)

        i = 8 + num_rows * 8 + start_pos
        for buf_len in lens:
            buf = contents[i:i+buf_len]
            i += buf_len
            if fn is not None:
                yield fn(buf)
            else:
                yield buf

    def _load_all(self, fn=None):
        table_descriptor = self._table._descriptor
        total_rows = table_descriptor.num_rows
        rows_per_item = table_descriptor.rows_per_item

        # Integer divide, round up
        num_items = int(math.ceil(total_rows / float(rows_per_item)))
        bufs = []
        input_rows = self._table.rows()
        assert len(input_rows) == total_rows
        i = 0
        for item_id in range(num_items):
            rows = total_rows % rows_per_item \
                   if item_id == num_items - 1 else rows_per_item
            for output in self._load_output_file(item_id, range(rows), fn):
                yield (input_rows[i], output)
                i += 1

    def _decode_png(self, png):
        return cv2.imdecode(np.frombuffer(png, dtype=np.dtype(np.uint8)),
                            cv2.IMREAD_COLOR)

    def load(self, fn=None):
        """
        Loads the results of a Scanner computation into Python.

        Kwargs:
            fn: Optional function to apply to the binary blobs as they are read
                in.

        Returns:
            Generator that yields either a numpy array for frame columns or
            a binary blob for non-frame columns (optionally processed by the
            `fn`).
        """

        # If the column is a video, then dump the requested frames to disk as PNGs
        # and return the decoded PNGs
        if self._descriptor.type == self._db._metadata_types.Video:
            sampler = self._db.sampler()
            tasks = sampler.all([(self._table.name(), '__scanner_png_dump')])
            [out_tbl] = self._db.run(tasks, self._db.evaluators.ImageEncoder())
            return out_tbl.columns(0).load(self._decode_png)
        else:
            return self._load_all(fn)

class Table:
    """
    A table in a Database.

    Can be part of many Collection objects.
    """
    def __init__(self, db, descriptor):
        self._db = db
        self._descriptor = descriptor
        job_id = self._descriptor.job_id
        if job_id != -1:
            self._job = self._db._load_descriptor(
                self._db._metadata_types.JobDescriptor,
                'jobs/{}/descriptor.bin'.format(job_id))
            self._task = None
            for task in self._job.tasks:
                if task.output_table_name == self._descriptor.name:
                    self._task = task
            if self._task is None:
                raise ScannerException('Table {} not found in job {}' \
                                   .format(self._descriptor.name, job_id))
        else:
            self._job = None

    def name(self):
        return self._descriptor.name

    def columns(self, index=None):
        columns = [Column(self, c) for c in self._descriptor.columns]
        return columns[index] if index is not None else columns

    def num_rows(self):
        return self._descriptor.num_rows

    def rows(self):
        if self._job is None:
            return list(range(self.num_rows()))
        else:
            if len(self._task.samples) == 1:
                return list(self._task.samples[0].rows)
            else:
                return list(range(self.num_rows()))

class Profiler:
    """
    Contains profiling information about Scanner jobs.
    """

    def __init__(self, db, job_id):
        self._storage = db._storage
        job = db._load_descriptor(
            db._metadata_types.JobDescriptor,
            'jobs/{}/descriptor.bin'.format(job_id))

        self._profilers = {}
        for n in range(job.num_nodes):
            path = '{}/jobs/{}/profile_{}.bin'.format(db._db_path, job_id, n)
            time, profs = self._parse_profiler_file(path)
            self._profilers[n] = (time, profs)

    def write_trace(self, path):
        """
        Generates a trace file in Chrome format.

        To visualize the trace, visit [chrome://tracing](chrome://tracing) in
        Google Chrome and click "Load" in the top left to load the trace.

        Args:
            path: Output path to write the trace.
        """

        traces = []
        next_tid = 0
        for proc, (_, worker_profiler_groups) in self._profilers.iteritems():
            for worker_type, profs in [('load', worker_profiler_groups['load']),
                                       ('decode', worker_profiler_groups['decode']),
                                       ('eval', worker_profiler_groups['eval']),
                                       ('save', worker_profiler_groups['save'])]:
                for i, prof in enumerate(profs):
                    tid = next_tid
                    next_tid += 1
                    worker_num = prof['worker_num']
                    tag = prof['worker_tag']
                    traces.append({
                        'name': 'thread_name',
                        'ph': 'M',
                        'pid': proc,
                        'tid': tid,
                        'args': {
                            'name': '{}_{:02d}_{:02d}'.format(
                                worker_type, proc, worker_num) + (
                                    "_" + str(tag) if tag else "")
                        }})
                    for interval in prof['intervals']:
                        traces.append({
                            'name': interval[0],
                            'cat': worker_type,
                            'ph': 'X',
                            'ts': interval[1] / 1000,  # ns to microseconds
                            'dur': (interval[2] - interval[1]) / 1000,
                            'pid': proc,
                            'tid': tid,
                            'args': {}
                        })
        with open(path, 'w') as f:
            f.write(json.dumps(traces))

    def _convert_time(self, d):
        def convert(t):
            return '{:2f}'.format(t / 1.0e9)
        return {k: self._convert_time(v) if isinstance(v, dict) else convert(v) \
                for (k, v) in d.iteritems()}

    def statistics(self):
        totals = {}
        for _, profiler in self._profilers.values():
            for kind in profiler:
                if not kind in totals: totals[kind] = {}
                for thread in profiler[kind]:
                    for (key, start, end) in thread['intervals']:
                        if not key in totals[kind]: totals[kind][key] = 0
                        totals[kind][key] += end-start

        readable_totals = self._convert_time(totals)
        return readable_totals

    def _read_advance(self, fmt, buf, offset):
        new_offset = offset + struct.calcsize(fmt)
        return struct.unpack_from(fmt, buf, offset), new_offset

    def _unpack_string(self, buf, offset):
        s = ''
        while True:
            t, offset = self._read_advance('B', buf, offset)
            c = t[0]
            if c == 0:
                break
            s += str(chr(c))
        return s, offset

    def _parse_profiler_output(self, bytes_buffer, offset):
        # Node
        t, offset = self._read_advance('q', bytes_buffer, offset)
        node = t[0]
        # Worker type name
        worker_type, offset = self._unpack_string(bytes_buffer, offset)
        # Worker tag
        worker_tag, offset = self._unpack_string(bytes_buffer, offset)
        # Worker number
        t, offset = self._read_advance('q', bytes_buffer, offset)
        worker_num = t[0]
        # Number of keys
        t, offset = self._read_advance('q', bytes_buffer, offset)
        num_keys = t[0]
        # Key dictionary encoding
        key_dictionary = {}
        for i in range(num_keys):
            key_name, offset = self._unpack_string(bytes_buffer, offset)
            t, offset = self._read_advance('B', bytes_buffer, offset)
            key_index = t[0]
            key_dictionary[key_index] = key_name
        # Intervals
        t, offset = self._read_advance('q', bytes_buffer, offset)
        num_intervals = t[0]
        intervals = []
        for i in range(num_intervals):
            # Key index
            t, offset = self._read_advance('B', bytes_buffer, offset)
            key_index = t[0]
            t, offset = self._read_advance('q', bytes_buffer, offset)
            start = t[0]
            t, offset = self._read_advance('q', bytes_buffer, offset)
            end = t[0]
            intervals.append((key_dictionary[key_index], start, end))
        # Counters
        t, offset = self._read_advance('q', bytes_buffer, offset)
        num_counters = t[0]
        counters = {}
        for i in range(num_counters):
            # Counter name
            counter_name, offset = self._unpack_string(bytes_buffer, offset)
            # Counter value
            t, offset = self._read_advance('q', bytes_buffer, offset)
            counter_value = t[0]
            counters[counter_name] = counter_value

        return {
            'node': node,
            'worker_type': worker_type,
            'worker_tag': worker_tag,
            'worker_num': worker_num,
            'intervals': intervals,
            'counters': counters
        }, offset

    def _parse_profiler_file(self, profiler_path):
        bytes_buffer = self._storage.read(profiler_path)
        offset = 0
        # Read start and end time intervals
        t, offset = self._read_advance('q', bytes_buffer, offset)
        start_time = t[0]
        t, offset = self._read_advance('q', bytes_buffer, offset)
        end_time = t[0]
        # Profilers
        profilers = defaultdict(list)
        # Load worker profilers
        t, offset = self._read_advance('B', bytes_buffer, offset)
        num_load_workers = t[0]
        for i in range(num_load_workers):
            prof, offset = self._parse_profiler_output(bytes_buffer, offset)
            profilers[prof['worker_type']].append(prof)
        # Eval worker profilers
        t, offset = self._read_advance('B', bytes_buffer, offset)
        num_eval_workers = t[0]
        t, offset = self._read_advance('B', bytes_buffer, offset)
        groups_per_chain = t[0]
        for pu in range(num_eval_workers):
            for fg in range(groups_per_chain):
                prof, offset = self._parse_profiler_output(bytes_buffer, offset)
                profilers[prof['worker_type']].append(prof)
        # Save worker profilers
        t, offset = self._read_advance('B', bytes_buffer, offset)
        num_save_workers = t[0]
        for i in range(num_save_workers):
            prof, offset = self._parse_profiler_output(bytes_buffer, offset)
            profilers[prof['worker_type']].append(prof)
        return (start_time, end_time), profilers

class NetDescriptor:
    def __init__(self, db):
        self._descriptor = db._arg_types[0].NetDescriptor()

    def _val(self, dct, key, default):
        if key in dct:
            return dct[key]
        else:
            return default

    @classmethod
    def from_file(cls, db, path):
        self = cls(db)
        with open(path) as f:
            args = toml.loads(f.read())

        d = self._descriptor
        net = args['net']
        d.model_path = net['model']
        d.model_weights_path = net['weights']
        d.input_layer_names.extend(net['input_layers'])
        d.output_layer_names.extend(net['output_layers'])
        d.input_width = self._val(net, 'input_width', -1)
        d.input_height = self._val(net, 'input_height', -1)
        d.normalize = self._val(net, 'normalize', False)
        d.preserve_aspect_ratio = self._val(net, 'preserve_aspect_ratio', False)
        d.transpose = self._val(net, 'tranpose', False)
        d.pad_mod = self._val(net, 'pad_mod', -1)

        mean = args['mean-image']
        if 'colors' in mean:
            order = net['input']['channel_ordering']
            for color in order:
                d.mean_colors.append(mean['colors'][color])
        elif 'image' in mean:
            d.mean_width = mean['width']
            d.mean_height = mean['height']
            # TODO: load mean binaryproto
            raise ScannerException('TODO')

        return self

    def as_proto(self):
        return self._descriptor
