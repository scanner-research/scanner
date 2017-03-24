import os
import os.path
import sys
import grpc
import imp
import socket
import time
import ipaddress
import pickle
import struct
from subprocess import Popen, PIPE
from random import choice
from string import ascii_uppercase
# Scanner imports
from common import *
from profiler import Profiler
from config import Config
from op import OpGenerator, Op
from sampler import Sampler
from collection import Collection
from table import Table
from column import Column


def start_master(port=None, config=None, config_path=None, block=False):
    """
    Start a master server instance on this node.

    Kwargs:
        config: A scanner Config object. If specified, config_path is
                ignored.
        config_path: Path to a Scanner configuration TOML, by default
                     assumed to be `~/.scanner.toml`.
        block: If true, will wait until the server is shutdown. Server
               will not shutdown currently unless wait_For_server_shutdown
               is eventually called.

    Returns:
        A cpp database instance.
    """
    config = config or Config(config_path)
    port = port or config.master_port

    # Load all protobuf types
    import libscanner as bindings
    db = bindings.Database(
        config.storage_config,
        config.db_path,
        config.master_address + ':' + port)
    result = bindings.start_master(db, port)
    if not result.success:
        raise ScannerException('Failed to start master: {}'.format(result.msg))
    if block:
        bindings.wait_for_server_shutdown(db)
    return db


def start_worker(master_address, port=None, config=None, config_path=None,
                 block=False):
    """
    Start a worker instance on this node.

    Args:
        master_address: The address of the master server to connect this worker
                        to.

    Kwargs:
        config: A scanner Config object. If specified, config_path is
                ignored.
        config_path: Path to a Scanner configuration TOML, by default
                     assumed to be `~/.scanner.toml`.
        block: If true, will wait until the server is shutdown. Server
               will not shutdown currently unless wait_ror_server_shutdown
               is eventually called.

    Returns:
        A cpp database instance.
    """
    config = config or Config(config_path)
    port = port or config.worker_port

    # Load all protobuf types
    import libscanner as bindings
    db = bindings.Database(
        config.storage_config,
        config.db_path,
        master_address)
    machine_params = bindings.default_machine_params()
    result = bindings.start_worker(db, machine_params, port)
    if not result.success:
        raise ScannerException('Failed to start worker: {}'.format(result.msg))
    if block:
        bindings.wait_for_server_shutdown(db)
    return result


class Database:
    """
    Entrypoint for all Scanner operations.

    Attributes:
        config: The Config object for the database.
        ops: An OpGenerator object for computation creation.
        protobufs: TODO(wcrichto)
    """

    def __init__(self, master=None, workers=None,
                 config_path=None, config=None,
                 debug=False):
        """
        Initializes a Scanner database.

        This will create a database at the `db_path` specified in the config
        if none exists.

        Kwargs:
            config_path: Path to a Scanner configuration TOML, by default
                         assumed to be `~/.scanner.toml`.
            config: A scanner Config object. If specified, config_path is
                    ignored.

        Returns:
            A database instance.
        """
        if config:
            self.config = config
        else:
            self.config = Config(config_path)

        self._debug = debug

        # Load all protobuf types
        import scanner.metadata_pb2 as metadata_types
        import scanner.engine.rpc_pb2 as rpc_types
        import scanner.types_pb2 as misc_types
        import libscanner as bindings

        self._protobufs = [misc_types, rpc_types, metadata_types]
        self._bindings = bindings

        # Setup database metadata
        self._db_path = self.config.db_path
        self._storage = self.config.storage
        self._cached_db_metadata = None
        self._png_dump_prefix = '__png_dump_'

        self.ops = OpGenerator(self)
        self.protobufs = ProtobufGenerator(self)

        self.start_cluster(master, workers)

        # Initialize database if it does not exist
        pydb_path = '{}/pydb'.format(self._db_path)
        if not os.path.isdir(pydb_path):
            os.mkdir(pydb_path)
            self._collections = self.protobufs.CollectionsDescriptor()
            self._update_collections()

        # Load database descriptors from disk
        self._collections = self._load_descriptor(
            self.protobufs.CollectionsDescriptor,
            'pydb/descriptor.bin')

    def __del__(self):
        self.stop_cluster()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, exception_tb):
        self.stop_cluster()

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
        include_dirs.append(self.config.module_dir + "/include")
        include_dirs.append(self.config.module_dir + "/build")
        flags = '{include} -std=c++11 -fPIC -shared -L{libdir} -lscanner {other}'
        return flags.format(
            include=" ".join(["-I " + d for d in include_dirs]),
            libdir='{}/build'.format(self.config.module_dir),
            other=self._bindings.other_flags())

    def print_build_flags(self):
        sys.stdout.write(self.get_build_flags())

    def summarize(self):
        summary = ''
        db_meta = self._load_db_metadata()
        tables = [
            ('TABLES', [
                ('Name', [t.name for t in db_meta.tables]),
                ('# rows', [
                    str(self.table(t.id).num_rows()) for t in db_meta.tables
                ]),
                ('Columns', [
                    ', '.join([c.name() for c in self.table(t.id).columns()])
                    for t in db_meta.tables
                ]),
            ]),
        ]

        if len(self._collections.names) > 0:
            tables.append(('COLLECTIONS', [
                ('Name', self._collections.names),
                ('# tables', [
                    str(len(self.collection(id).table_names()))
                    for id in self._collections.ids
                ])
            ]))

        for table_idx, (label, cols) in enumerate(tables):
            if table_idx > 0:
                summary += '\n\n'
            num_cols = len(cols)
            max_col_lens = [max(max([len(s) for s in c]), len(name))
                            for name, c in cols]
            table_width = sum(max_col_lens) + 3*(num_cols-1)
            label = '** {} **'.format(label)
            summary += ' ' * (table_width/2 - len(label)/2) + label + '\n'
            summary += '-'*table_width + '\n'
            col_name_fmt = ' | '.join(['{{:{}}}' for _ in range(num_cols)])
            col_name_fmt = col_name_fmt.format(*max_col_lens)
            summary += col_name_fmt.format(*[s for s, _ in cols]) + '\n'
            summary += '-'*table_width + '\n'
            row_fmt = ' | '.join(['{{:{}}}' for _ in range(num_cols)])
            row_fmt = row_fmt.format(*max_col_lens)
            for i in range(len(cols[0][1])):
                summary += row_fmt.format(*[c[i] for _, c in cols]) + '\n'
        return summary

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
                self.protobufs.DatabaseDescriptor,
                'db_metadata.bin')
            self._cached_db_metadata = desc
        return self._cached_db_metadata

    def _connect_to_master(self):
        channel = grpc.insecure_channel(
            self._master_address,
            options=[('grpc.max_message_length', 24499183 * 2)])
        self._master = self.protobufs.MasterStub(channel)
        result = False
        try:
            self._master.Ping(self.protobufs.Empty())
            result = True
        except grpc.RpcError as e:
            status = e.code()
            if status == grpc.StatusCode.UNAVAILABLE:
                pass
            elif status == grpc.StatusCode.OK:
                result = True
            else:
                raise ScannerException('Master ping errored with status: {}'
                                       .format(status))
        return result

    def _run_remote_cmd(self, host, cmd):
        host_ip, _, _ = host.partition(':')
        host_ip = unicode(socket.gethostbyname(host_ip), "utf-8")
        if ipaddress.ip_address(host_ip).is_loopback:
            return Popen(cmd, shell=True)
        else:
            cmd = cmd.replace('"', '\\"')
            return Popen("ssh {} \"cd {} && {}\"".format(host_ip, os.getcwd(), cmd), shell=True)

    def start_cluster(self, master, workers):
        """
        Starts  a Scanner cluster.

        Args:
            master: ssh-able address of the master node.
            workers: list of ssh-able addresses of the worker nodes.
        """

        if master is None:
            self._master_address = (
                self.config.master_address + ':' + self.config.master_port)
        else:
            self._master_address = master
        if workers is None:
            self._worker_addresses = [
                self.config.master_address + ':' + self.config.worker_port]
        else:
            self._worker_addresses = workers

        # Boot up C++ database bindings
        self._db = self._bindings.Database(
            self.config.storage_config,
            self._db_path,
            self._master_address)

        if self._debug:
            self._master_conn = None
            self._worker_conns = None
            machine_params = self._bindings.default_machine_params()
            res = self._bindings.start_master(
                self._db, self.config.master_port).success
            assert res
            res = self._connect_to_master()
            assert res
            for i in range(len(self._worker_addresses)):
                res = self._bindings.start_worker(
                    self._db, machine_params,
                    str(int(self.config.worker_port) + i)).success
                assert res
        else:
            pickled_config = pickle.dumps(self.config)
            master_cmd = (
                'python -c ' +
                '\"from scannerpy import start_master\n' +
                'import pickle\n' +
                'config=pickle.loads(\'\'\'{config:s}\'\'\')\n' +
                'start_master(port=\'{master_port:s}\', block=True, config=config)\"').format(
                    master_port=self.config.master_port,
                    config=pickled_config)
            worker_cmd = (
                'python -c ' +
                '\"from scannerpy import start_worker\n' +
                'import pickle\n' +
                'config=pickle.loads(\'\'\'{config:s}\'\'\')\n' +
                'start_worker(\'{master:s}\', port=\'{worker_port:s}\', block=True, config=config)\"')
            self._master_conn = self._run_remote_cmd(self._master_address,
                                                     master_cmd)

            # Wait for master to start
            slept_so_far = 0
            sleep_time = 20
            while slept_so_far < sleep_time:
                if self._connect_to_master():
                    break
                time.sleep(0.3)
                slept_so_far += 0.3
            if slept_so_far >= sleep_time:
                self._master_conn.kill()
                self._master_conn = None
                raise ScannerException('Timed out waiting to connect to master')

            # Start workers now that master is ready
            self._worker_conns = [
                self._run_remote_cmd(w, worker_cmd.format(
                    master=self._master_address,
                    config=pickled_config,
                    worker_port=w.partition(':')[2]))
                for w in self._worker_addresses]
            slept_so_far = 0
            sleep_time = 20
            while slept_so_far < sleep_time:
                active_workers = self._master.ActiveWorkers(self.protobufs.Empty())
                if (len(active_workers.workers) > len(self._worker_addresses)):
                    raise ScannerException(
                        ('Master has more workers than requested ' +
                         '({:d} vs {:d})').format(len(active_workers.workers),
                                                  len(self._worker_addresses)))
                if (len(active_workers.workers) == len(self._worker_addresses)):
                    break
                time.sleep(0.3)
                slept_so_far += 0.3
            if slept_so_far >= sleep_time:
                self._master_conn.wait()
                for wc in self._worker_conns:
                    wc.wait()
                self._master_conn = None
                self._worker_conns = None
                raise ScannerException(
                    'Timed out waiting for workers to connect to master')

        # Load stdlib
        stdlib_path = '{}/build/stdlib'.format(self.config.module_dir)
        self.load_op('{}/libstdlib.so'.format(stdlib_path),
                     '{}/stdlib_pb2.py'.format(stdlib_path))

    def stop_cluster(self):
        if self._master:
            self._try_rpc(
                lambda: self._master.Shutdown(self.protobufs.Empty()))
            self._master = None
        if self._master_conn:
            self._master_conn.kill()
            self._master_conn = None
        if self._worker_conns:
            for wc in self._worker_conns:
                wc.kill()
            self._worker_conns = None

    def _try_rpc(self, fn):
        try:
            result = fn()
        except grpc.RpcError as e:
            raise ScannerException(e)

        if isinstance(result, self.protobufs.Result):
            if not result.success:
                raise ScannerException(result.msg)

        return result

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
            mod = imp.load_source('_ignore', proto_path)
            self._protobufs.append(mod)
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
        collection = self.protobufs.CollectionDescriptor()
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
            (list of created Tables, list of (path, reason) failures to ingest)
        """

        if len(videos) == 0:
            raise ScannerException('Must ingest at least one video.')

        [table_names, paths] = zip(*videos)
        for table_name in table_names:
            if self.has_table(table_name):
                if force is True:
                    self._delete_table(table_name)
                else:
                    raise ScannerException(
                        'Attempted to ingest over existing table {}'
                        .format(table_name))
        self._save_descriptor(self._load_db_metadata(), 'db_metadata.bin')
        ingest_params = self.protobufs.IngestParameters()
        ingest_params.table_names.extend(table_names)
        ingest_params.video_paths.extend(paths)
        ingest_result = self._try_rpc(
            lambda: self._master.IngestVideos(ingest_params))
        if not ingest_result.result.success:
            raise ScannerException(ingest_result.result.msg)
        failures = zip(ingest_result.failed_paths, ingest_result.failed_messages)

        self._cached_db_metadata = None
        return ([self.table(t) for (t, p) in videos
                if p not in ingest_result.failed_paths],
                failures)

    def ingest_video_collection(self, collection_name, videos, force=False):
        """
        Creates a Collection from a list of videos.

        Args:
            collection_name: String name of the Collection to create.
            videos: List of video paths.

        Kwargs:
            force: TODO(wcrichto)

        Returns:
            (Collection, list of (path, reason) failures to ingest)
        """
        table_names = ['{}:{:03d}'.format(collection_name, i)
                       for i in range(len(videos))]
        tables, failures = self.ingest_videos(zip(table_names, videos), force)
        collection = self.new_collection(
            collection_name, [t.name() for t in tables], force)
        return collection, failures

    def has_collection(self, name):
        return name in self._collections.names

    def collection(self, name):
        if isinstance(name, basestring):
            index = self._collections.names[:].index(name)
            id = self._collections.ids[index]
        else:
            id = name
        collection = self._load_descriptor(
            self.protobufs.CollectionDescriptor,
            'pydb/collection_{}.bin'.format(id))
        return Collection(self, name, collection)

    def has_table(self, name):
        db_meta = self._load_db_metadata()
        for table in db_meta.tables:
            if table.name == name:
                return True
        return False

    def _delete_table(self, name):
        table = self.table(name)
        db_meta = self._load_db_metadata()
        for i, t in enumerate(db_meta.tables):
            if t.id == table.id():
                del db_meta.tables[i]
                return
        assert False

    def delete_table(self, name):
        self._delete_table(name)
        self._save_descriptor(db_meta, 'db_metadata.bin')

    def new_table(self, name, columns, rows, force=False):
        if self.has_table(name):
            if force:
                #self.delete_table(name)
                pass
            else:
                raise ScannerException('Attempted to create table with existing '
                                       'name {}'.format(name))
        columns.insert(0, "index")
        for i, row in enumerate(rows):
            row.insert(0, struct.pack('=Q', i))
        self._bindings.new_table(self._db, name, columns, rows)
        self._cached_db_metadata = None

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
            self.protobufs.TableDescriptor,
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

        eval_sorted[-1]._inputs.insert(0, (eval_sorted[0], ["index"]))

        return [e.to_proto(eval_index) for e in eval_sorted]

    def _get_op_output_info(self, op_name):
        op_output_info_args = self.protobufs.OpOutputInfoArgs()
        op_output_info_args.op_name = op_name

        op_output_info = self._try_rpc (lambda: self._master.GetOpOutputInfo(op_output_info_args))

        if not op_output_info.result.success:
            raise ScannerException(op_output_info.result.msg)

        return op_output_info

    def _check_has_op(self, op_name):
        self._get_op_output_info(op_name)

    def _get_output_columns(self, op_name):
        return self._get_op_output_info(op_name).output_columns

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
                    out_cols = self._get_output_columns(op[i]._name)
                op[i+1]._inputs = [(op[i], out_cols)]
            op = op[-1]

        # If the user doesn't explicitly specify an OutputTable, assume that
        # it's all the output columns of the last op.
        if op._name != "OutputTable":
            out_cols = self._get_output_columns(str(op._name))
            op = Op.output(self, [(op, out_cols)])

        return self._toposort(op)

    def _parse_size_string(self, s):
        (prefix, suffix) = (s[:-1], s[-1])
        mults = {
            'G': 1024**3,
            'M': 1024**2,
            'K': 1024**1
        }
        if suffix not in mults:
            raise ScannerException('Invalid size suffix in "{}"'.format(s))
        return int(prefix) * mults[suffix]

    def run(self, tasks, op,
            output_collection=None,
            job_name=None,
            force=False,
            work_item_size=250,
            cpu_pool=None,
            gpu_pool=None,
            pipeline_instances_per_node=-1,
            show_progress=True):
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
            work_item_size: TODO(wcrichto)
            cpu_pool: TODO(wcrichto)
            gpu_pool: TODO(wcrichto)
            pipeline_instances_per_node: TODO(wcrichto)
            show_progress: TODO(wcrichto)

        Returns:
            Either the output Collection if output_collection is specified
            or a list of Table objects.
        """

        # If the input is a collection, assume user is running over all frames
        input_is_collection = isinstance(tasks, Collection)
        if input_is_collection:
            if output_collection is None:
                raise ScannerException(
                    'If Database.run input is a collection, output_collection_name '
                    'must be specified')
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
                    self._delete_table(task.output_table_name)
                else:
                    raise ScannerException('Job would overwrite existing table {}'
                                           .format(task.output_table_name))
        self._save_descriptor(self._load_db_metadata(), 'db_metadata.bin')

        job_params = self.protobufs.JobParameters()
        # Generate a random job name if none given
        job_name = job_name or ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.task_set.tasks.extend(tasks)
        job_params.task_set.ops.extend(self._process_dag(op))
        job_params.pipeline_instances_per_node = pipeline_instances_per_node
        job_params.work_item_size = work_item_size
        job_params.show_progress = show_progress

        job_params.memory_pool_config.pinned_cpu = False
        if cpu_pool is not None:
            job_params.memory_pool_config.cpu.use_pool = True
            if cpu_pool[0] == 'p':
                job_params.memory_pool_config.pinned_cpu = True
                cpu_pool = cpu_pool[1:]
            size = self._parse_size_string(cpu_pool)
            job_params.memory_pool_config.cpu.free_space = size

        if gpu_pool is not None:
            job_params.memory_pool_config.gpu.use_pool = True
            size = self._parse_size_string(gpu_pool)
            job_params.memory_pool_config.gpu.free_space = size

        # Run the job
        self._try_rpc(lambda: self._master.NewJob(job_params))

        # Invalidate db metadata because of job run
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
        for mod in self._db._protobufs:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise ScannerException('No protobuf with name {}'.format(name))
