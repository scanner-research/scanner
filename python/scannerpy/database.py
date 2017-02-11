import os
import os.path
import sys
import grpc
import importlib
import socket
from subprocess import Popen, PIPE
from random import choice
from string import ascii_uppercase

from common import *
from profiler import Profiler
from config import Config
from op import OpGenerator, Op
from sampler import Sampler
from collection import Collection
from table import Table
from column import Column


class Database:
    """
    Entrypoint for all Scanner operations.

    Attributes:
        config: The Config object for the database.
        ops: An OpGenerator object for computation creation.
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
        import scanner.kernels.types_pb2 as misc_types
        import scanner_bindings as bindings
        self._metadata_types = metadata_types
        self._rpc_types = rpc_types
        self._arg_types = [arg_types, misc_types, rpc_types]
        self._bindings = bindings

        # Setup database metadata
        self._db_path = self.config.db_path
        self._storage = self.config.storage
        self._master_address = self.config.master_address
        self._cached_db_metadata = None

        self.ops = OpGenerator(self)
        self.protobufs = ProtobufGenerator(self)

        # Initialize database if it does not exist
        pydb_path = '{}/pydb'.format(self._db_path)
        self._db = self._bindings.Database(
            self.config.storage_config,
            self._db_path,
            self._master_address)
        if not os.path.isdir(pydb_path):
            os.mkdir(pydb_path)
            self._collections = self._metadata_types.CollectionsDescriptor()
            self._update_collections()

        # Load database descriptors from disk
        self._collections = self._load_descriptor(
            self._metadata_types.CollectionsDescriptor,
            'pydb/descriptor.bin')

    def get_build_flags(self):
        """
        Gets the g++ build flags for compiling custom ops.

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
        if self._cached_db_metadata is None:
            desc = self._load_descriptor(
                self._metadata_types.DatabaseDescriptor,
                'db_metadata.bin')
            self._cached_db_metadata = desc
        return self._cached_db_metadata

    def _connect_to_master(self):
        channel = grpc.insecure_channel(self._master_address)
        self._master = self._rpc_types.MasterStub(channel)

        # Ping master and start master/worker locally if they don't exist.
        try:
            self._master.Ping(self._rpc_types.Empty())
        except grpc.RpcError as e:
            status = e.code()
            if status == grpc.StatusCode.UNAVAILABLE:
                log.info("Master not started, creating temporary master/worker...")
                # If they get GC'd then the masters/workers will die, so persist
                # them until the database object dies
                self._ignore1 = self.start_master()
                self._ignore2 = self.start_worker()
                log.info("Temporary master/worker started")

                # If we don't reconnect to master, there's a 5-10 sec delay for
                # for original connection to reboot
                channel = grpc.insecure_channel(self._master_address)
                self._master = self._rpc_types.MasterStub(channel)
            elif status == grpc.StatusCode.OK:
                pass
            else:
                raise ScannerException('Master ping errored with status: {}'
                                       .format(status))


    def start_master(self):
        """
        TODO(wcrichto)
        """

        return self._bindings.start_master(self._db)

    def start_worker(self):
        """
        TODO(wcrichto)
        """

        machine_params = self._bindings.default_machine_params()
        return self._bindings.start_worker(self._db, machine_params)

    def _run_remote_cmd(self, host, cmd):
        local_ip = socket.gethostbyname(socket.gethostname())
        if socket.gethostbyname(host) == local_ip:
            return Popen(cmd, shell=True)
        else:
            print "ssh {} {}".format(host, cmd)
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

    def _try_rpc(self, fn):
        try:
            result = fn()
        except grpc.RpcError as e:
            raise ScannerException(e)

        if not result.success:
            raise ScannerException(result.msg)


    def load_op(self, so_path, proto_path=None):
        """
        Loads a custom op into the Scanner runtime.

        By convention, if the op requires arguments from Python, it must
        have a protobuf message called <OpName>Args, e.g. BlurArgs or
        HistogramArgs, and the path to that protobuf should be provided.

        Args:
            so_path: Path to the custom op's shared object file.

        Kwargs:
            proto_path: Path to the custom op's arguments protobuf
                        if one exists.
        """
        if proto_path is not None:
            if not os.path.isfile(proto_path):
                raise ScannerException('Protobuf path does not exist: {}'
                                       .format(proto_path))
            (proto_dir, mod_file) = os.path.split(proto_path)
            sys.path.append(proto_dir)
            (mod_name, _) = os.path.splitext(mod_file)
            self._arg_types.append(importlib.import_module(mod_name))
        self._connect_to_master()
        op_info = self.protobufs.OpInfo()
        op_info.so_path = so_path
        self._try_rpc(lambda: self._master.LoadOp(op_info))

    def _update_collections(self):
        self._save_descriptor(self._collections, 'pydb/descriptor.bin')

    def delete_collection(self, collection_name):
        if collection_name not in self._collections.names:
            raise ScannerException('Collection with name {} does not exist'
                                   .format(collection_name))

        index = self._collections.names[:].index(collection_name)
        id = self._collections.ids[index]
        del self._collections.names[index]
        del self._collections.ids[index]

        os.remove('{}/pydb/collection_{}.bin'.format(self._db_path, id))

    def new_collection(self, collection_name, table_names, force=False, job_id=None):
        """
        Creates a new Collection from a list of tables.

        Args:
            collection_name: String name of the collection to create.
            table_names: List of table name strings to put in the collection.

        Kwargs:
            force: TODO(wcrichto)
            job_id: TODO(wcrichto)

        Returns:
            The new Collection object.
        """

        if collection_name in self._collections.names:
            if force:
                self.delete_collection(collection_name)
            else:
                raise ScannerException(
                    'Collection with name {} already exists'
                    .format(collection_name))

        last_id = self._collections.ids[-1] if len(self._collections.ids) > 0 else -1
        new_id = last_id + 1
        self._collections.ids.append(new_id)
        self._collections.names.append(collection_name)
        self._update_collections()
        collection = self._metadata_types.CollectionDescriptor()
        collection.tables.extend(table_names)
        collection.job_id = -1 if job_id is None else job_id
        self._save_descriptor(collection, 'pydb/collection_{}.bin'.format(new_id))

        return self.collection(collection_name)

    def ingest_videos(self, videos, force=False):
        """
        Creates a Table from a video.

        Args:
            videos: TODO(wcrichto)


        Kwargs:
            force: TODO(wcrichto)

        Returns:
            The newly created Table object.
        """

        [table_names, paths] = zip(*videos)
        for table_name in table_names:
            if self.has_table(table_name):
                if force is True:
                    self.delete_table(table_name)
                else:
                    raise ScannerException(
                        'Attempted to ingest over existing table {}'
                        .format(table_name))
        invalid = self._bindings.ingest_videos(
            self._db,
            list(table_names),
            list(paths))
        invalid = [i.path for i in invalid]
        self._cached_db_metadata = None
        return [self.table(t) for (t, p) in videos if p not in invalid]

    def ingest_video_collection(self, collection_name, videos, force=False):
        """
        Creates a Collection from a list of videos.

        Args:
            collection_name: String name of the Collection to create.
            videos: List of video paths.

        Kwargs:
            force: TODO(wcrichto)

        Returns:
            The newly created Collection object.
        """
        table_names = ['{}:{:03d}'.format(collection_name, i)
                       for i in range(len(videos))]
        tables = self.ingest_videos(zip(table_names, videos), force)
        collection = self.new_collection(
            collection_name, [t.name() for t in tables], force)
        return collection

    def has_collection(self, name):
        return name in self._collections.names

    def collection(self, name):
        index = self._collections.names[:].index(name)
        id = self._collections.ids[index]
        collection = self._load_descriptor(
            self._metadata_types.CollectionDescriptor,
            'pydb/collection_{}.bin'.format(id))
        return Collection(self, name, collection)

    def has_table(self, name):
        db_meta = self._load_db_metadata()
        for table in db_meta.tables:
            if table.name == name:
                return True
        return False

    def delete_table(self, name):
        table = self.table(name)
        db_meta = self._load_db_metadata()
        for i, t in enumerate(db_meta.tables):
            if t.id == table.id():
                del db_meta.tables[i]
                self._save_descriptor(db_meta, 'db_metadata.bin')
                return
        assert False

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
            for table in db_meta.tables:
                if table.name == name and table.id != table_id:
                    raise ScannerException(
                        'Internal error: multiple tables with same name: {}'.format(name))
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

    def _toposort(self, op):
        edges = defaultdict(list)
        in_edges_left = defaultdict(int)
        start_node = None

        explored_nodes = set()
        stack = [op]
        while len(stack) > 0:
            c = stack.pop()
            explored_nodes.add(c)
            if (c._name == "InputTable"):
                start_node = c
                continue
            elif len(c._inputs) == 0:
                input = Op.input(self)
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

    def _process_dag(self, op):
        # If ops are passed as a list (e.g. [transform, caffe])
        # then hook up inputs to outputs of adjacent ops
        if isinstance(op, list):
            for i in range(len(op) - 1):
                if len(op[i+1]._inputs) > 0:
                    continue
                if op[i]._name == "InputTable":
                    out_cols = ["frame", "frame_info"]
                else:
                    out_cols = self._bindings.get_output_columns(op[i]._name)
                op[i+1]._inputs = [(op[i], out_cols)]
            op = op[-1]

        # If the user doesn't explicitly specify an OutputTable, assume that
        # it's all the output columns of the last op.
        if op._name != "OutputTable":
            out_cols = self._bindings.get_output_columns(str(op._name))
            op = Op.output(self, [(op, out_cols)])

        return self._toposort(op)

    def run(self, tasks, op, output_collection=None, job_name=None,
            force=False, io_item_size=1000, work_item_size=250):
        """
        Runs a computation over a set of inputs.

        Args:
            tasks: The set of inputs to run the computation on. If tasks is a
                   Collection, then the computation is run on all frames of all
                   tables in the collection. Otherwise, tasks should be generated
                   by the Sampler.
            op: The computation to run. Op is either a list of
                   ops to run in sequence, or a DAG with the output node
                   passed in as the argument.

        Kwargs:
            output_collection: If this is not None, then a new collection with
                               this name will be created for all the output
                               tables.
            job_name: An optional name to assign the job. It will be randomly
                      generated if none is given.
            force: TODO(wcrichto)

        Returns:
            Either the output Collection if output_collection is specified
            or a list of Table objects.
        """

        # If the input is a collection, assume user is running over all frames
        input_is_collection = isinstance(tasks, Collection)
        if input_is_collection:
            sampler = self.sampler()
            tasks = sampler.all(tasks)

        # If the output should be a collection, then set the table names
        if output_collection is not None:
            if self.has_collection(output_collection) and not force:
                raise ScannerException(
                    'Collection with name {} already exists'
                    .format(output_collection))
            for task in tasks:
                new_name = '{}:{}'.format(
                    output_collection,
                    task.samples[0].table_name.split(':')[-1])
                task.output_table_name = new_name

        for task in tasks:
            if self.has_table(task.output_table_name):
                if force:
                    self.delete_table(task.output_table_name)
                else:
                    raise ScannerException('Job would overwrite existing table {}'
                                           .format(task.output_table_name))

        job_params = self._rpc_types.JobParameters()
        # Generate a random job name if none given
        job_name = job_name or ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.task_set.tasks.extend(tasks)
        job_params.task_set.ops.extend(self._process_dag(op))
        job_params.kernel_instances_per_node = self.config.kernel_instances_per_node
        job_params.io_item_size = io_item_size
        job_params.work_item_size = work_item_size

        # Execute job via RPC
        self._try_rpc(lambda: self._master.NewJob(job_params))

        self._cached_db_metadata = None

        db_meta = self._load_db_metadata()
        job_id = None
        for job in db_meta.jobs:
            if job.name == job_name:
                job_id = job.id
        if job_id is None:
            raise ScannerException('Internal error: job id not found after run')

        # Return a new collection if the input was a collection, otherwise
        # return a table list
        table_names = [task.output_table_name for task in tasks]
        if output_collection is not None:
            return self.new_collection(output_collection, table_names, force, job_id)
        else:
            return [self.table(t) for t in table_names]


class ProtobufGenerator:
    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        for mod in self._db._arg_types:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise ScannerException('No protobuf with name {}'.format(name))
