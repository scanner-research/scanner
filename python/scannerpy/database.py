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
import signal
import copy
import collections

from timeit import default_timer as now
from multiprocessing import Process, Queue
from subprocess import Popen, PIPE
from random import choice
from string import ascii_uppercase
from threading import Thread
from multiprocessing.dummy import Pool as ThreadPool

from common import *
from profiler import Profiler
from config import Config
from op import OpGenerator, Op, OpColumn
from sampler import Sampler
from partitioner import TaskPartitioner
from collection import Collection
from table import Table
from column import Column
from protobuf_generator import ProtobufGenerator

from storehousepy import StorageConfig, StorageBackend
from job import Job
from bulk_job import BulkJob

def start_master(port=None, config=None, config_path=None, block=False, watchdog=True):
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
    result = bindings.start_master(db, port, watchdog)
    if not result.success:
        raise ScannerException('Failed to start master: {}'.format(result.msg))
    if block:
        bindings.wait_for_server_shutdown(db)
    return db


def start_worker(master_address, machine_params=None, port=None, config=None,
                 config_path=None, block=False, watchdog=True):
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
        #storage_config,
        config.db_path,
        master_address)
    machine_params = machine_params or bindings.default_machine_params()
    result = bindings.start_worker(db, machine_params, str(port), watchdog)
    if not result.success:
        raise ScannerException('Failed to start worker: {}'.format(result.msg))
    if block:
        bindings.wait_for_server_shutdown(db)
    return result


class Database(object):
    """
    Entrypoint for all Scanner operations.

    Attributes:
        config: The Config object for the database.
        ops: An OpGenerator object for computation creation.
        protobufs: TODO(wcrichto)
    """

    def __init__(self, master=None, workers=None,
                 config_path=None, config=None,
                 debug=None, start_cluster=True):
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

        self._start_cluster = start_cluster
        self._debug = debug
        if debug is None:
            self._debug = (master is None and workers is None)

        self._master = None

        import libscanner as bindings
        self._bindings = bindings

        # Setup database metadata
        self._db_path = self.config.db_path
        self._storage = self.config.storage
        self._cached_db_metadata = None
        self._png_dump_prefix = '__png_dump_{:s}'

        self.ops = OpGenerator(self)
        self.sampler = Sampler(self)
        self.partitioner = TaskPartitioner(self)
        self.protobufs = ProtobufGenerator(self.config)
        self._op_cache = {}

        self._workers = {}
        self.start_cluster(master, workers);

        # Initialize database if it does not exist
        pydb_path = '{}/pydb'.format(self._db_path)

        pydbpath_info = self._storage.get_file_info(pydb_path+'/')

        if not (pydbpath_info.file_exists and pydbpath_info.file_is_folder):
            self._storage.make_dir(pydb_path)
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
        del self._db

    def summarize(self):
        summary = ''
        db_meta = self._load_db_metadata()
        if len(db_meta.tables) == 0:
            return 'Your database is empty!'

        tables = [
            ('TABLES', [
                ('Name', [t.name for t in db_meta.tables]),
                ('# rows', [
                    str(self.table(t.id).num_rows()) for t in db_meta.tables
                ]),
                ('Columns', [
                    ', '.join(self.table(t.id).column_names())
                    for t in db_meta.tables
                ]),
                ('Committed', ['true' if self.table(t.id).committed()
                               else 'false'
                               for t in db_meta.tables]),
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
            max_col_lens = [max(max([len(s) for s in c] or [0]), len(name))
                            for name, c in cols]
            table_width = sum(max_col_lens) + 3*(num_cols-1)
            label = '** {} **'.format(label)
            summary += ' ' * (table_width/2 - len(label)/2) + label + '\n'
            summary += '-' * table_width + '\n'
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
        d.ParseFromString(
            self._storage.read('{}/{}'.format(self._db_path, path)))
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
            # table id cache
            self._table_id = {}
            self._table_name = {}
            self._table_committed = {}
            for i, table in enumerate(self._cached_db_metadata.tables):
                if table.name in self._table_name:
                    raise ScannerException(
                        'Internal error: multiple tables with same name: {}'.format(name))
                self._table_id[table.id] = i
                self._table_name[table.name] = i
                self._table_committed[table.id] = table.committed
        return self._cached_db_metadata

    def _connect_to_worker(self, address):
        channel = grpc.insecure_channel(
            address,
            options=[('grpc.max_message_length', 24499183 * 2)])
        worker = self.protobufs.WorkerStub(channel)
        try:
            self._master.Ping(self.protobufs.Empty())
            return worker
        except grpc.RpcError as e:
            status = e.code()
            if status == grpc.StatusCode.UNAVAILABLE:
                pass
            else:
                raise ScannerException('Master ping errored with status: {}'
                                       .format(status))
        return None


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

    def _start_heartbeat(self):
        # Start up heartbeat to keep master alive
        def heartbeat_task(q, master_address):
            import scanner.metadata_pb2 as metadata_types
            import scanner.engine.rpc_pb2 as rpc_types
            import scanner.types_pb2 as misc_types
            import libscanner as bindings

            channel = grpc.insecure_channel(
                master_address,
                options=[('grpc.max_message_length', 24499183 * 2)])
            master = rpc_types.MasterStub(channel)
            while q.empty():
                master.PokeWatchdog(rpc_types.Empty())
                time.sleep(1)

        self._heartbeat_queue = Queue()
        self._heartbeat_process = Process(target=heartbeat_task,
                                          args=(self._heartbeat_queue,
                                                self._master_address))
        self._heartbeat_process.daemon = True
        self._heartbeat_process.start()

    def _stop_heartbeat(self):
        self._heartbeat_queue.put(0)

    def _handle_signal(self, signum, frame):
        if (signum == signal.SIGINT or
            signum == signal.SIGTERM or
            signum == signal.SIGKILL):
            # Stop cluster
            self._stop_heartbeat()
            self.stop_cluster()
            sys.exit(1)

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
            str(self._db_path),
            str(self._master_address))

        if self._start_cluster:
            # Set handler to shutdown cluster on signal
            # TODO(apoms): we should clear these handlers when stopping
            # the cluster
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

            if self._debug:
                self._master_conn = None
                self._worker_conns = None
                machine_params = self._bindings.default_machine_params()
                res = self._bindings.start_master(
                    self._db, self.config.master_port, True).success
                assert res
                res = self._connect_to_master()
                assert res

                self._start_heartbeat()

                for i in range(len(self._worker_addresses)):
                    res = self._bindings.start_worker(
                        self._db, machine_params,
                        str(int(self.config.worker_port) + i), True).success
                    assert res
            else:
                master_port = self._master_address.partition(':')[2]
                pickled_config = pickle.dumps(self.config)
                master_cmd = (
                    'python -c ' +
                    '\"from scannerpy import start_master\n' +
                    'import pickle\n' +
                    'config=pickle.loads(\'\'\'{config:s}\'\'\')\n' +
                    'start_master(port=\'{master_port:s}\', block=True, config=config)\"').format(
                        master_port=master_port,
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
                # Start up heartbeat to keep master alive
                self._start_heartbeat()

                # Start workers now that master is ready
                self._worker_conns = [
                    self._run_remote_cmd(w, worker_cmd.format(
                        master=self._master_address,
                        config=pickled_config,
                        worker_port=w.partition(':')[2]))
                    for w in self._worker_addresses]
                slept_so_far = 0
                # Has to be this long for GCS
                sleep_time = 60
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
                    self._master_conn.kill()
                    for wc in self._worker_conns:
                        wc.kill()
                    self._master_conn = None
                    self._worker_conns = None
                    raise ScannerException(
                        'Timed out waiting for workers to connect to master')
        else:
            self._master_conn = None
            self._worker_conns = None

            # Wait for master to start
            slept_so_far = 0
            sleep_time = 20
            while slept_so_far < sleep_time:
                if self._connect_to_master():
                    break
                time.sleep(0.3)
                slept_so_far += 0.3
            if slept_so_far >= sleep_time:
                raise ScannerException('Timed out waiting to connect to master')

        # Load stdlib
        stdlib_path = '{}/build/stdlib'.format(self.config.module_dir)
        self.load_op('{}/libstdlib.so'.format(stdlib_path),
                     '{}/stdlib_pb2.py'.format(stdlib_path))

    def stop_cluster(self):
        if self._start_cluster:
           if self._master:
               # Stop heartbeat
               self._stop_heartbeat()
               try:
                   self._try_rpc(
                       lambda: self._master.Shutdown(self.protobufs.Empty()))
               except:
                   pass
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
            self.protobufs.add_module(proto_path)
        op_path = self.protobufs.OpPath()
        op_path.path = so_path
        self._try_rpc(lambda: self._master.LoadOp(op_path))

    def register_op(self, name, input_columns, output_columns,
                    variadic_inputs=False, stencil=None, proto_path=None):
        op_registration = self.protobufs.OpRegistration()
        op_registration.name = name
        op_registration.variadic_inputs = variadic_inputs

        def add_col(columns, col):
            if isinstance(col, basestring):
                c = columns.add()
                c.name = col
                c.type = self.protobufs.Other
            elif isinstance(col, collections.Iterable):
                c = columns.add()
                c.name = col[0]
                c.type = ColumnType.to_proto(self.protobufs, col[1])
            else:
                raise ScannerException(
                    'Column ' + col + ' must be a string name or a tuple of '
                    '(name, column_type)')
        for in_col in input_columns:
            add_col(op_registration.input_columns, in_col)
        for out_col in output_columns:
            add_col(op_registration.output_columns, out_col)
        if stencil is None:
            op_registration.can_stencil = False
        else:
            op_registration.can_stencil = True
            op_registration.preferred_stencil.extend(stencil)
        if proto_path is not None:
            self.protobufs.add_module(proto_path)
        self._try_rpc(lambda: self._master.RegisterOp(op_registration))

    def register_python_kernel(self, op_name, device_type, kernel_path, 
                               batch=1):
        with open(kernel_path, 'r') as f:
            kernel_str = f.read()
        py_registration = self.protobufs.PythonKernelRegistration()
        py_registration.op_name = op_name
        py_registration.device_type = DeviceType.to_proto(self.protobufs,
                                                          device_type)
        py_registration.kernel_str = kernel_str
        py_registration.pickled_config = pickle.dumps(self.config)
        py_registration.batch_size = batch
        self._try_rpc(
            lambda: self._master.RegisterPythonKernel(py_registration))

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

        self._storage.delete_file('{}/pydb/collection_{}.bin'.format(self._db_path, id))

    def new_collection(self, collection_name, tables, force=False, job_id=None):
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

        table_names = []
        for table in tables:
            table_names.append(table.name())

        last_id = (
            self._collections.ids[-1] if len(self._collections.ids) > 0 else -1)
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
        to_delete = []
        for table_name in table_names:
            if self.has_table(table_name):
                if force is True:
                    to_delete.append(table_name)
                else:
                    raise ScannerException(
                        'Attempted to ingest over existing table {}'
                        .format(table_name))
        self.delete_tables(to_delete)
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
            collection_name, tables, force)
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
        if name in self._table_name:
            return True
        return False

    def delete_tables(self, names):
        db_meta = self._load_db_metadata()
        idxs_to_delete = []
        for name in names:
            assert name in self._table_name
            t = self.table(name)
            t._need_descriptor()
            idxs_to_delete.append(self._table_name[name])
        idxs_to_delete.sort()
        for idx in reversed(idxs_to_delete):
            del db_meta.tables[idx]
        self._save_descriptor(db_meta, 'db_metadata.bin')
        self._cached_db_metadata = None
        self._load_db_metadata()

    def delete_table(self, name):
        self.delete_tables([name])

    def new_table(self, name, columns, rows, fn=None, force=False):
        """
        Creates a new table from a list of rows.

        Args:
            name: String name of the table to create
            columns: List of names of table columns
            rows: List of rows with each row a list of elements corresponding
                  to the specified columns. Elements must be strings of
                  serialized representations of the data.

        Kwargs:
            fn: TODO(wcrichto)
            force: TODO(apoms)

        Returns:
            The new table object.
        """

        if self.has_table(name):
            if force:
                self.delete_table(name)
            else:
                raise ScannerException('Attempted to create table with existing '
                                       'name {}'.format(name))
        if fn is not None:
            rows = [fn(row, self.protobufs) for row in rows]
        cols = copy.copy(columns)
        cols.insert(0, "index")
        for i, row in enumerate(rows):
            row.insert(0, struct.pack('=Q', i))
        self._bindings.new_table(self._db, name, cols, rows)
        self._cached_db_metadata = None
        return self.table(name)

    def table(self, name):
        db_meta = self._load_db_metadata()

        table_name = None
        table_id = None
        if isinstance(name, basestring):
            if name in self._table_name:
                table_name = name
                table_id = db_meta.tables[self._table_name[name]].id
            if table_id is None:
                raise ScannerException('Table with name {} not found'.format(name))
        elif isinstance(name, int):
            if name in self._table_id:
                table_id = name
                table_name = db_meta.tables[self._table_id[name]].name
            if table_id is None:
                raise ScannerException('Table with id {} not found'.format(name))
        else:
            raise ScannerException('Invalid table identifier')

        return Table(self, table_name, table_id)

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

    def _get_op_info(self, op_name):
        if op_name in self._op_cache:
            op_info = self._op_cache[op_name]
        else:
            op_info_args = self.protobufs.OpInfoArgs()
            op_info_args.op_name = op_name

            op_info = self._try_rpc(lambda: self._master.GetOpInfo(op_info_args))

            if not op_info.result.success:
                raise ScannerException(op_info.result.msg)

            self._op_cache[op_name] = op_info

        return op_info

    def _check_has_op(self, op_name):
        self._get_op_info(op_name)

    def _get_input_columns(self, op_name):
        return self._get_op_info(op_name).input_columns

    def _get_output_columns(self, op_name):
        return self._get_op_info(op_name).output_columns

    def _toposort(self, dag):
        op = dag
        # Perform DFS on modified graph
        edges = defaultdict(list)
        in_edges_left = defaultdict(int)

        input_nodes = []
        explored_nodes = set()
        stack = [op]
        while len(stack) > 0:
            c = stack.pop()
            if c in explored_nodes:
                continue
            explored_nodes.add(c)

            if c._name == "Input":
                input_nodes.append(c)
                continue

            for input in c._inputs:
                edges[input._op].append(c)
                in_edges_left[c] += 1

                if input._op not in explored_nodes:
                    stack.append(input._op)

        # Keep track of position of input ops and sampling/slicing ops
        # to use for associating job args to
        input_ops = {}
        sampling_slicing_ops = {}
        output_ops = {}

        # Compute sorted list
        eval_sorted = []
        eval_index = {}
        stack = input_nodes[:]
        while len(stack) > 0:
            c = stack.pop()
            eval_sorted.append(c)
            op_idx = len(eval_sorted) - 1
            eval_index[c] = op_idx
            for child in edges[c]:
                in_edges_left[child] -= 1
                if in_edges_left[child] == 0:
                    stack.append(child)
            if c._name == "Input":
                input_ops[c] = op_idx
            elif (c._name == "Sample" or
                  c._name == "Space" or
                  c._name == "Slice" or
                  c._name == "Unslice"):
                sampling_slicing_ops[c] = op_idx
            elif c._name == "OutputTable":
                output_ops[c] = op_idx

        return [e.to_proto(eval_index) for e in eval_sorted], \
            input_ops, \
            sampling_slicing_ops, \
            output_ops

    def _parse_size_string(self, s):
        (prefix, suffix) = (s[:-1], s[-1])
        mults = {
            'G': 1024**3,
            'M': 1024**2,
            'K': 1024**1
        }
        suffix = suffix.upper()
        if suffix not in mults:
            raise ScannerException('Invalid size suffix in "{}"'.format(s))
        return int(prefix) * mults[suffix]

    def run(self, bulk_job,
            force=False,
            work_packet_size=250,
            io_packet_size=-1,
            cpu_pool=None,
            gpu_pool=None,
            pipeline_instances_per_node=None,
            show_progress=True,
            profiling=False,
            load_sparsity_threshold=8,
            tasks_in_queue_per_pu=4):
        """
        Runs a computation over a set of inputs.

        Args:
            tasks: The set of inputs to run the computation on. If tasks is a
                   Collection, then the computation is run on all frames of all
                   tables in the collection. Otherwise, tasks should be generated
                   by the Sampler.
            outputs: TODO(wcrichto)

        Kwargs:
            output_collection: If this is not None, then a new collection with
                               this name will be created for all the output
                               tables.
            force: TODO(wcrichto)
            work_item_size: TODO(wcrichto)
            io_item_size: TODO(wcrichto)
            cpu_pool: TODO(wcrichto)
            gpu_pool: TODO(wcrichto)
            pipeline_instances_per_node: TODO(wcrichto)
            show_progress: TODO(wcrichto)

        Returns:
            Either the output Collection if output_collection is specified
            or a list of Table objects.
        """
        assert isinstance(bulk_job, BulkJob)
        assert isinstance(bulk_job.output(), Op)

        # Collect compression annotations to add to job
        compression_options = []
        output_op = bulk_job.output()
        for out_col in output_op.inputs():
            opts = self.protobufs.OutputColumnCompression()
            opts.codec = 'default'
            if out_col._type == self.protobufs.Video:
                for k, v in out_col._encode_options.iteritems():
                    if k == 'codec':
                        opts.codec = v
                    else:
                        opts.options[k] = str(v)
            compression_options.append(opts)

        sorted_ops, input_ops, sampling_slicing_ops, output_ops = (
            self._toposort(bulk_job.output()))

        job_params = self.protobufs.BulkJobParameters()
        job_name = ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.ops.extend(sorted_ops)
        job_output_table_names = []
        for job in bulk_job.jobs():
            j = job_params.jobs.add()
            output_table_name = None
            for op_col, args in job.op_args().iteritems():
                if isinstance(op_col, Op):
                    op = op_col
                else:
                    op = op_col._op
                if op in input_ops:
                    op_idx = input_ops[op]
                    col_input = j.inputs.add()
                    col_input.op_index = op_idx
                    if not args._table.committed():
                        raise ScannerException(
                            'Attempted to bind table {name} to Input Op but '
                            'table {name} is not committed.'
                            .format(name=args._table.name()))
                    col_input.table_name = args._table.name()
                    col_input.column_name = args.name()
                elif op in sampling_slicing_ops:
                    op_idx = sampling_slicing_ops[op]
                    saa = j.sampling_args_assignment.add()
                    saa.op_index = op_idx
                    if not isinstance(args, list):
                        args = [args]
                    for arg in args:
                        sa = saa.sampling_args.add()
                        sa.CopyFrom(arg)
                elif op in output_ops:
                    op_idx = output_ops[op]
                    assert isinstance(args, basestring)
                    output_table_name = args
                    job_output_table_names.append(args)
                else:
                    raise ScannerException(
                        'Attempted to bind arguments to Op {} which is not '
                        'an input, sampling, spacing, slicing, or output Op.'
                        .format(op.name()))  # FIXME(apoms): op.name() is unbound
            if output_table_name is None:
                raise ScannerException(
                    'Did not specify the output table name by binding a '
                    'string to the output Op.')
            j.output_table_name = output_table_name

        # Delete tables if they exist and force was specified
        to_delete = []
        for name in job_output_table_names:
            if self.has_table(name):
                if force:
                    to_delete.append(name)
                else:
                    raise ScannerException(
                        'Job would overwrite existing table {}'
                        .format(name))
        self.delete_tables(to_delete)

        job_params.compression.extend(compression_options)
        job_params.pipeline_instances_per_node = (
            pipeline_instances_per_node or -1)
        job_params.work_packet_size = work_packet_size
        job_params.io_packet_size = io_packet_size
        job_params.show_progress = show_progress
        job_params.profiling = profiling
        job_params.tasks_in_queue_per_pu = tasks_in_queue_per_pu
        job_params.load_sparsity_threshold = load_sparsity_threshold
        job_params.boundary_condition = (
            self.protobufs.BulkJobParameters.REPEAT_EDGE)

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

        while True:
            try:
                result = self._master.IsJobDone(self.protobufs.Empty())
            except grpc.RpcError as e:
                raise ScannerException(e)
            if result.finished:
                break
            else:
                time.sleep(1.0)

        if not result.result.success:
            raise ScannerException(result.result.msg)

        # Invalidate db metadata because of job run
        self._cached_db_metadata = None

        db_meta = self._load_db_metadata()
        job_id = None
        for job in db_meta.bulk_jobs:
            if job.name == job_name:
                job_id = job.id
        if job_id is None:
            raise ScannerException('Internal error: job id not found after run')

        return [self.table(t) for t in job_output_table_names]
