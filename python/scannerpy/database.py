from __future__ import absolute_import, division, print_function, unicode_literals
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
import subprocess
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue, cpu_count
from subprocess import Popen, PIPE
from random import choice
from string import ascii_uppercase

from scannerpy.common import *
from scannerpy.profiler import Profiler
from scannerpy.config import Config
from scannerpy.op import OpGenerator, Op, OpColumn
from scannerpy.source import SourceGenerator, Source
from scannerpy.sink import SinkGenerator, Sink
from scannerpy.sampler import Sampler
from scannerpy.partitioner import TaskPartitioner
from scannerpy.table import Table
from scannerpy.column import Column
from scannerpy.protobuf_generator import ProtobufGenerator, python_to_proto
from scannerpy.job import Job
from scannerpy.bulk_job import BulkJob

from storehouse import StorageConfig, StorageBackend

import scannerpy.libscanner as bindings
import scanner.metadata_pb2 as metadata_types
import scanner.engine.rpc_pb2 as rpc_types
import scanner.engine.rpc_pb2_grpc as grpc_types
import scanner.types_pb2 as misc_types

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def start_master(port=None,
                 config=None,
                 config_path=None,
                 block=False,
                 watchdog=True,
                 prefetch_table_metadata=True,
                 no_workers_timeout=30):
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
    db = bindings.Database(
        config.storage_config, config.db_path.encode('ascii'),
        (config.master_address + ':' + port).encode('ascii'))
    result = bindings.start_master(db, port.encode('ascii'), watchdog,
                                   prefetch_table_metadata, no_workers_timeout)
    if not result.success():
        raise ScannerException('Failed to start master: {}'.format(
            result.msg()))
    if block:
        bindings.wait_for_server_shutdown(db)
    return db


def worker_process((master_address, machine_params, port, config, config_path,
                    block, watchdog, prefetch_table_metadata)):
    config = config or Config(config_path)
    port = port or config.worker_port

    # Load all protobuf types
    db = bindings.Database(
        config.storage_config,
        #storage_config,
        config.db_path.encode('ascii'),
        master_address.encode('ascii'))
    machine_params = machine_params or bindings.default_machine_params()
    result = bindings.start_worker(db, machine_params,
                                   str(port).encode('ascii'), watchdog,
                                   prefetch_table_metadata)
    if not result.success():
        raise ScannerException('Failed to start worker: {}'.format(
            result.msg()))
    if block:
        bindings.wait_for_server_shutdown(db)
    return result


def start_worker(master_address,
                 machine_params=None,
                 port=None,
                 config=None,
                 config_path=None,
                 block=False,
                 watchdog=True,
                 prefetch_table_metadata=True,
                 num_workers=None):
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

    if num_workers is not None:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                executor.map(
                    worker_process,
                    ([(master_address, machine_params, int(port) + i, config,
                       config_path, block, watchdog, prefetch_table_metadata)
                      for i in range(num_workers)])))

        for result in results:
            if not result.success: return result
        return results[0]

    else:
        return worker_process(
            (master_address, machine_params, port, config, config_path, block,
             watchdog, prefetch_table_metadata))


class Database(object):
    """
    Entrypoint for all Scanner operations.

    Attributes:
        config: The Config object for the database.
        ops: An OpGenerator object for computation creation.
        protobufs: TODO(wcrichto)
    """

    def __init__(self,
                 master=None,
                 workers=None,
                 config_path=None,
                 config=None,
                 debug=None,
                 start_cluster=True,
                 prefetch_table_metadata=True,
                 no_workers_timeout=30,
                 grpc_timeout=30):
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
        self._prefetch_table_metadata = prefetch_table_metadata
        self._no_workers_timeout = no_workers_timeout
        self._debug = debug
        self._grpc_timeout = grpc_timeout
        if debug is None:
            self._debug = (master is None and workers is None)

        self._master = None

        self._bindings = bindings

        # Setup database metadata
        self._db_path = self.config.db_path
        self._storage = self.config.storage
        self._cached_db_metadata = None
        self._png_dump_prefix = '__png_dump_{:s}'

        self.ops = OpGenerator(self)
        self.sources = SourceGenerator(self)
        self.sinks = SinkGenerator(self)
        self.sampler = Sampler(self)
        self.partitioner = TaskPartitioner(self)
        self.protobufs = ProtobufGenerator(self.config)
        self._op_cache = {}

        self._workers = {}
        self._worker_conns = None
        self.start_cluster(master, workers)

    def __del__(self):
        self.stop_cluster()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, exception_tb):
        self.stop_cluster()
        del self._db

    def has_gpu(self):
        try:
            with open(os.devnull, 'w') as f:
                subprocess.check_call(['nvidia-smi'], stdout=f, stderr=f)
            return True
        except:
            pass
        return False

    def summarize(self):
        summary = ''
        db_meta = self._load_db_metadata()
        if len(db_meta.tables) == 0:
            return 'Your database is empty!'

        tables = [
            ('TABLES', [
                ('ID', [str(t.id) for t in db_meta.tables]),
                ('Name', [t.name for t in db_meta.tables]),
                ('# rows',
                 [str(self.table(t.id).num_rows()) for t in db_meta.tables]),
                ('Columns', [
                    ', '.join(self.table(t.id).column_names())
                    for t in db_meta.tables
                ]),
                ('Committed', [
                    'true' if self.table(t.id).committed() else 'false'
                    for t in db_meta.tables
                ]),
            ]),
        ]

        for table_idx, (label, cols) in enumerate(tables):
            if table_idx > 0:
                summary += '\n\n'
            num_cols = len(cols)
            max_col_lens = [
                max(max([len(s) for s in c] or [0]), len(name))
                for name, c in cols
            ]
            table_width = sum(max_col_lens) + 3 * (num_cols - 1)
            label = '** {} **'.format(label)
            summary += ' ' * int(
                table_width / 2 - len(label) / 2) + label + '\n'
            summary += '-' * table_width + '\n'
            col_name_fmt = ' | '.join(['{{:{}}}' for _ in range(num_cols)])
            col_name_fmt = col_name_fmt.format(*max_col_lens)
            summary += col_name_fmt.format(*[s for s, _ in cols]) + '\n'
            summary += '-' * table_width + '\n'
            row_fmt = ' | '.join(['{{:{}}}' for _ in range(num_cols)])
            row_fmt = row_fmt.format(*max_col_lens)
            for i in range(len(cols[0][1])):
                summary += row_fmt.format(*[c[i] for _, c in cols]) + '\n'
        return summary

    def _load_descriptor(self, descriptor, path):
        d = descriptor()
        path = '{}/{}'.format(self._db_path, path)
        try:
            d.ParseFromString(self._storage.read(path.encode('ascii')))
        except UserWarning:
            raise ScannerException(
                'Internal error. Missing file {}'.format(path))
        return d

    def _save_descriptor(self, descriptor, path):
        self._storage.write(('{}/{}'.format(self._db_path,
                                            path)).encode('ascii'),
                            descriptor.SerializeToString())

    def _load_table_metadata(self, table_names):
        NUM_TABLES_TO_READ = 100000
        tables = []
        for i in range(0, len(table_names), NUM_TABLES_TO_READ):
            get_tables_params = self.protobufs.GetTablesParams()
            for table_name in table_names[i:i + NUM_TABLES_TO_READ]:
                get_tables_params.tables.append(table_name)
            get_tables_result = self._try_rpc(
                lambda: self._master.GetTables(get_tables_params))
            if not get_tables_result.result.success:
                raise ScannerException(
                    'Internal error: GetTables returned error: {}'.format(
                        get_tables_result.result.msg))
            tables.extend(get_tables_result.tables)
        return tables

    def _load_db_metadata(self):
        if self._cached_db_metadata is None:
            desc = self._load_descriptor(self.protobufs.DatabaseDescriptor,
                                         'db_metadata.bin')
            self._cached_db_metadata = desc
            # table id cache
            self._table_id = {}
            self._table_name = {}
            self._table_committed = {}
            for i, table in enumerate(self._cached_db_metadata.tables):
                if table.name in self._table_name:
                    raise ScannerException(
                        'Internal error: multiple tables with same name: {}'.
                        format(table.name))
                self._table_id[table.id] = i
                self._table_name[table.name] = i
                self._table_committed[table.id] = table.committed

            if self._prefetch_table_metadata:
                self._table_descriptor = {}
                # Read all table descriptors from database
                table_names = self._table_name.keys()
                tables = self._load_table_metadata(table_names)
                for table in tables:
                    self._table_descriptor[table.id] = table

        return self._cached_db_metadata

    def _make_grpc_channel(self, address):
        max_message_length = 1024 * 1024 * 1024
        return grpc.insecure_channel(
            address,
            options=[('grpc.max_send_message_length', max_message_length),
                     ('grpc.max_receive_message_length', max_message_length)])

    def _connect_to_worker(self, address):
        channel = self._make_grpc_channel(address)
        worker = self.protobufs.WorkerStub(channel)
        try:
            self._worker.Ping(self.protobufs.Empty(),
                              timeout=self._grpc_timeout)
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
        channel = self._make_grpc_channel(self._master_address)
        self._master = self.protobufs.MasterStub(channel)
        result = False
        try:
            self._master.Ping(self.protobufs.Empty(),
                              timeout=self._grpc_timeout)
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

    def _run_remote_cmd(self, host, cmd, nohup=False):
        host_name, _, _ = host.partition(':')
        host_ip = unicode(socket.gethostbyname(host_name), "utf-8")
        if ipaddress.ip_address(host_ip).is_loopback:
            return Popen(cmd, shell=True)
        else:
            cmd = cmd.replace('"', '\\"')
            return Popen(
                "ssh {} \"cd {} && {} {} {}\"".format(host_name, os.getcwd(),
                                                      ''
                                                      if nohup else '', cmd, ''
                                                      if nohup else ''),
                shell=True)

    def _start_heartbeat(self):
        # Start up heartbeat to keep master alive
        def heartbeat_task(q, master_address):
            channel = self._make_grpc_channel(master_address)
            master = grpc_types.MasterStub(channel)
            while q.empty():
                try:
                    master.PokeWatchdog(rpc_types.Empty(),
                                        timeout=self._grpc_timeout)
                except grpc.RpcError as e:
                    pass
                time.sleep(1)

        self._heartbeat_queue = Queue()
        self._heartbeat_process = Process(
            target=heartbeat_task,
            args=(self._heartbeat_queue, self._master_address))
        self._heartbeat_process.daemon = True
        self._heartbeat_process.start()

    def _stop_heartbeat(self):
        self._heartbeat_queue.put(0)

    def _handle_signal(self, signum, frame):
        if (signum == signal.SIGINT or signum == signal.SIGTERM
                or signum == signal.SIGKILL):
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
                self.config.master_address + ':' + self.config.worker_port
            ]
        else:
            self._worker_addresses = workers

        # Boot up C++ database bindings
        self._db = self._bindings.Database(
            self.config.storage_config,
            str(self._db_path).encode('ascii'),
            str(self._master_address).encode('ascii'))

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
                    self._db, self.config.master_port.encode('ascii'), True,
                    self._prefetch_table_metadata,
                    self._no_workers_timeout).success
                assert res
                res = self._connect_to_master()
                if not res:
                    raise ScannerException(
                        'Failed to connect to local master process on port '
                        '{:s}. (Is there another process that is bound to that '
                        'port already?)'.format(self.config.master_port))

                self._start_heartbeat()

                for i in range(len(self._worker_addresses)):
                    res = self._bindings.start_worker(
                        self._db, machine_params,
                        str(int(self.config.worker_port) + i).encode('ascii'),
                        True, self._prefetch_table_metadata).success
                    if not res:
                        raise ScannerException(
                            'Failed to start local worker on port {:d} and '
                            'connect to master. (Is there another process that '
                            'is bound to that port already?)'.format(
                                self.config.worker_port))
            else:
                master_port = self._master_address.partition(':')[2]
                pickled_config = pickle.dumps(self.config)
                master_cmd = (
                    'python -c ' + '\"from scannerpy import start_master\n' +
                    'import pickle\n' +
                    'config=pickle.loads(\'\'\'{config:s}\'\'\')\n' +
                    'start_master(port=\'{master_port:s}\', block=True,\n' +
                    '             config=config,\n' +
                    '             prefetch_table_metadata={prefetch},\n' +
                    '             no_workers_timeout={no_workers})\" ' +
                    '').format(
                        master_port=master_port,
                        config=pickled_config,
                        prefetch=self._prefetch_table_metadata,
                        no_workers=self._no_workers_timeout)
                worker_cmd = (
                    'python -c ' + '\"from scannerpy import start_worker\n' +
                    'import pickle\n' +
                    'config=pickle.loads(\'\'\'{config:s}\'\'\')\n' +
                    'start_worker(\'{master:s}\', port=\'{worker_port:s}\',\n'
                    + '             block=True,\n' +
                    '             config=config)\" ' + '')
                self._master_conn = self._run_remote_cmd(
                    self._master_address, master_cmd, nohup=True)

                # Wait for master to start
                slept_so_far = 0
                sleep_time = 60
                while slept_so_far < sleep_time:
                    if self._connect_to_master():
                        break
                    time.sleep(0.3)
                    slept_so_far += 0.3
                if slept_so_far >= sleep_time:
                    self._master_conn.kill()
                    self._master_conn = None
                    raise ScannerException(
                        'Timed out waiting to connect to master')
                # Start up heartbeat to keep master alive
                self._start_heartbeat()

                # Start workers now that master is ready
                self._worker_conns = []
                ignored_nodes = 0
                for w in self._worker_addresses:
                    try:
                        self._worker_conns.append(
                            self._run_remote_cmd(
                                w,
                                worker_cmd.format(
                                    master=self._master_address,
                                    config=pickled_config,
                                    worker_port=w.partition(':')[2]),
                                nohup=True))
                    except:
                        print('WARNING: Failed to ssh into {:s}, ignoring'.
                              format(w))
                        ignored_nodes += 1
                slept_so_far = 0
                # Has to be this long for GCS
                sleep_time = 60
                while slept_so_far < sleep_time:
                    active_workers = self._master.ActiveWorkers(
                        self.protobufs.Empty(), timeout=self._grpc_timeout)
                    if (len(active_workers.workers) > len(self._worker_conns)):
                        raise ScannerException(
                            ('Master has more workers than requested ' +
                             '({:d} vs {:d})').format(
                                 len(active_workers.workers),
                                 len(self._worker_conns)))
                    if (len(active_workers.workers) == len(
                            self._worker_conns)):
                        break
                    time.sleep(0.3)
                    slept_so_far += 0.3
                if slept_so_far >= sleep_time:
                    self.stop_cluster()
                    raise ScannerException(
                        'Timed out waiting for workers to connect to master')
                if ignored_nodes > 0:
                    print('Ignored {:d} nodes during startup.'.format(
                        ignored_nodes))
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
                raise ScannerException(
                    'Timed out waiting to connect to master')

        # Load stdlib
        self.load_op('{}/libstdlib.so'.format(SCRIPT_DIR),
                     '{}/../scanner/stdlib/stdlib_pb2.py'.format(SCRIPT_DIR))

    def stop_cluster(self):
        if self._start_cluster:
            if self._master:
                # Stop heartbeat
                self._stop_heartbeat()
                try:
                    self._try_rpc(
                        lambda: self._master.Shutdown(
                            self.protobufs.Empty(), timeout=self._grpc_timeout))
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
        self._try_rpc(lambda: self._master.LoadOp(
            op_path, timeout=self._grpc_timeout))

    def register_op(self,
                    name,
                    input_columns,
                    output_columns,
                    variadic_inputs=False,
                    stencil=None,
                    proto_path=None,
                    unbounded_state=False,
                    bounded_state=None):
        op_registration = self.protobufs.OpRegistration()
        op_registration.name = name
        op_registration.variadic_inputs = variadic_inputs
        op_registration.has_unbounded_state = unbounded_state

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

        if bounded_state is not None:
            assert isinstance(bounded_state, int)
            op_registration.has_bounded_state = True
            op_registration.warmup = bounded_state

        if proto_path is not None:
            self.protobufs.add_module(proto_path)

        self._try_rpc(lambda: self._master.RegisterOp(
            op_registration, timeout=self._grpc_timeout))

    def register_python_kernel(self,
                               op_name,
                               device_type,
                               kernel_path,
                               batch=1):
        with open(kernel_path, 'r') as f:
            kernel_str = f.read()
        py_registration = self.protobufs.PythonKernelRegistration()
        py_registration.op_name = op_name
        py_registration.device_type = DeviceType.to_proto(
            self.protobufs, device_type)
        py_registration.kernel_str = kernel_str
        py_registration.pickled_config = pickle.dumps(self.config)
        py_registration.batch_size = batch
        self._try_rpc(
            lambda: self._master.RegisterPythonKernel(
                py_registration, timeout=self._grpc_timeout))

    def ingest_videos(self, videos, inplace=False, force=False):
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
        ingest_params.inplace = inplace
        ingest_result = self._try_rpc(
            lambda: self._master.IngestVideos(ingest_params))
        if not ingest_result.result.success:
            raise ScannerException(ingest_result.result.msg)
        failures = zip(ingest_result.failed_paths,
                       ingest_result.failed_messages)

        self._cached_db_metadata = None
        return ([
            self.table(t) for (t, p) in videos
            if p not in ingest_result.failed_paths
        ], failures)

    def has_table(self, name):
        db_meta = self._load_db_metadata()
        if name in self._table_name:
            return True
        return False

    def delete_tables(self, names):
        delete_tables_params = self.protobufs.DeleteTablesParams()
        for name in names:
            delete_tables_params.tables.append(name)
        self._try_rpc(lambda: self._master.DeleteTables(delete_tables_params))
        self._cached_db_metadata = None

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
                raise ScannerException(
                    'Attempted to create table with existing '
                    'name {}'.format(name))
        if fn is not None:
            rows = [fn(row, self.protobufs) for row in rows]

        params = self.protobufs.NewTableParams()
        params.table_name = name
        params.columns[:] = ["index"] + columns

        for i, row in enumerate(rows):
            row_proto = params.rows.add()
            row_proto.columns[:] = [struct.pack('=Q', i)] + row

        self._try_rpc(lambda: self._master.NewTable(params))

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
                raise ScannerException(
                    'Table with name {} not found'.format(name))
        elif isinstance(name, int):
            if name in self._table_id:
                table_id = name
                table_name = db_meta.tables[self._table_id[name]].name
            if table_id is None:
                raise ScannerException(
                    'Table with id {} not found'.format(name))
        else:
            raise ScannerException('Invalid table identifier')

        table = Table(self, table_name, table_id)
        if self._prefetch_table_metadata:
            table._descriptor = self._table_descriptor[table_id]

        return table

    def profiler(self, job_name):
        db_meta = self._load_db_metadata()
        if isinstance(job_name, basestring):
            job_id = None
            for job in db_meta.jobs:
                if job.name == job_name:
                    job_id = job.id
                    break
            if job_id is None:
                raise ScannerException(
                    'Job name {} does not exist'.format(job_name))
        else:
            job_id = job_name

        return Profiler(self, job_id)

    def _get_source_info(self, source_name):
        #if op_name in self._op_cache:
        #    op_info = self._op_cache[op_name]
        #else:
        source_info_args = self.protobufs.SourceInfoArgs()
        source_info_args.source_name = source_name

        source_info = self._try_rpc(
            lambda: self._master.GetSourceInfo(source_info_args))

        if not source_info.result.success:
            raise ScannerException(source_info.result.msg)

        #self._op_cache[op_name] = op_info

        return source_info

    def _get_sink_info(self, sink_name):
        #if op_name in self._op_cache:
        #    op_info = self._op_cache[op_name]
        #else:
        sink_info_args = self.protobufs.SinkInfoArgs()
        sink_info_args.sink_name = sink_name

        sink_info = self._try_rpc(
            lambda: self._master.GetSinkInfo(sink_info_args))

        if not sink_info.result.success:
            raise ScannerException(sink_info.result.msg)

        #self._op_cache[op_name] = op_info

        return sink_info

    def _get_op_info(self, op_name):
        if op_name in self._op_cache:
            op_info = self._op_cache[op_name]
        else:
            op_info_args = self.protobufs.OpInfoArgs()
            op_info_args.op_name = op_name

            op_info = self._try_rpc(
                lambda: self._master.GetOpInfo(
                    op_info_args, self._grpc_timeout))

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

        source_nodes = []
        explored_nodes = set()
        stack = [op]
        while len(stack) > 0:
            c = stack.pop()
            if c in explored_nodes:
                continue
            explored_nodes.add(c)

            if isinstance(c, Source):
                source_nodes.append(c)
                continue

            for input in c._inputs:
                edges[input._op].append(c)
                in_edges_left[c] += 1

                if input._op not in explored_nodes:
                    stack.append(input._op)

        # Keep track of position of input ops and sampling/slicing ops
        # to use for associating job args to
        source_ops = {}
        sampling_slicing_ops = {}
        output_ops = {}

        # Compute sorted list
        eval_sorted = []
        eval_index = {}
        stack = source_nodes[:]
        while len(stack) > 0:
            c = stack.pop()
            eval_sorted.append(c)
            op_idx = len(eval_sorted) - 1
            eval_index[c] = op_idx
            for child in edges[c]:
                in_edges_left[child] -= 1
                if in_edges_left[child] == 0:
                    stack.append(child)
            if isinstance(c, Source):
                source_ops[c] = op_idx
            elif (c._name == "Sample" or c._name == "Space"
                  or c._name == "Slice" or c._name == "Unslice"):
                sampling_slicing_ops[c] = op_idx
            elif isinstance(c, Sink):
                output_ops[c] = op_idx

        return [e.to_proto(eval_index) for e in eval_sorted], \
            source_ops, \
            sampling_slicing_ops, \
            output_ops

    def _parse_size_string(self, s):
        (prefix, suffix) = (s[:-1], s[-1])
        mults = {'G': 1024**3, 'M': 1024**2, 'K': 1024**1}
        suffix = suffix.upper()
        if suffix not in mults:
            raise ScannerException('Invalid size suffix in "{}"'.format(s))
        return int(prefix) * mults[suffix]

    def wait_on_current_job(self, show_progress=True):
        pbar = None
        total_tasks = None
        last_task_count = 0
        last_jobs_failed = 0
        last_failed_workers = 0
        while True:
            try:
                job_status = self._master.GetJobStatus(
                    self.protobufs.Empty(), timeout=self._grpc_timeout)
                if show_progress and pbar is None and job_status.total_jobs != 0 \
                   and job_status.total_tasks != 0:
                    total_tasks = job_status.total_tasks
                    pbar = tqdm(total=total_tasks)
            except grpc.RpcError as e:
                raise ScannerException(e)
            if job_status.finished:
                break
            if pbar is not None:
                tasks_completed = job_status.tasks_done
                if tasks_completed - last_task_count > 0:
                    pbar.update(tasks_completed - last_task_count)
                last_task_count = tasks_completed
                pbar.set_postfix({
                    'jobs':
                    job_status.total_jobs - job_status.jobs_done,
                    'tasks':
                    job_status.total_tasks - job_status.tasks_done,
                    'workers':
                    job_status.num_workers,
                })
                time_str = time.strftime('%l:%M%p %z on %b %d, %Y')
                if last_jobs_failed < job_status.jobs_failed:
                    num_jobs_failed = job_status.jobs_failed - last_jobs_failed
                    pbar.write('{:d} {:s} failed at {:s}'.format(
                        num_jobs_failed, 'job'
                        if num_jobs < 2 else 'jobs', time_str))
                if last_failed_workers < job_status.failed_workers:
                    num_workers_failed = job_status.failed_workers - last_failed_workers
                    pbar.write('{:d} {:s} failed at {:s}'.format(
                        num_workers_failed, 'worker'
                        if num_workers_failed < 2 else 'workers', time_str))
                last_jobs_failed = job_status.jobs_failed
                last_failed_workers = job_status.failed_workers
            time.sleep(1.0)

        if pbar is not None:
            pbar.close()

        return job_status

    def bulk_fetch_video_metadata(self, tables):
        params = self.protobufs.GetVideoMetadataParams(
            tables=[t.name() for t in tables])
        result = self._try_rpc(lambda: self._master.GetVideoMetadata(
            params, timeout=self._grpc_timeout))
        return result.videos

    def run(self,
            bulk_job,
            force=False,
            work_packet_size=250,
            io_packet_size=-1,
            cpu_pool=None,
            gpu_pool=None,
            pipeline_instances_per_node=None,
            show_progress=True,
            profiling=False,
            load_sparsity_threshold=8,
            tasks_in_queue_per_pu=4,
            task_timeout=0,
            checkpoint_frequency=1000):
        assert isinstance(bulk_job, BulkJob)
        assert isinstance(bulk_job.output(), Sink)

        if (bulk_job.output()._name == 'FrameColumn' or
            bulk_job.output()._name == 'Column'):
            is_table_output = True
        else:
            is_table_output = False

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
        # Get output columns
        output_column_names = output_op._output_names

        sorted_ops, source_ops, sampling_slicing_ops, output_ops = (
            self._toposort(bulk_job.output()))

        job_params = self.protobufs.BulkJobParameters()
        job_name = ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.ops.extend(sorted_ops)
        job_output_table_names = []
        job_params.output_column_names[:] = output_column_names
        for job in bulk_job.jobs():
            j = job_params.jobs.add()
            output_table_name = None
            for op_col, args in job.op_args().iteritems():
                if isinstance(op_col, Op) or isinstance(op_col, Sink):
                    op = op_col
                else:
                    op = op_col._op
                if op in source_ops:
                    op_idx = source_ops[op]
                    source_input = j.inputs.add()
                    source_input.op_index = op_idx
                    # We special case on Column to transform it into a
                    # (table, col) pair that is then trasnformed into a
                    # protobuf object
                    if isinstance(args, Column):
                        if not args._table.committed():
                            raise ScannerException(
                                'Attempted to bind table {name} to Input Op '
                                'but table {name} is not committed.'.format(
                                    name=args._table.name()))
                        args = {'table_name': args._table.name(),
                                'column_name': args.name()}
                    n = op._name
                    if n.startswith('Frame'):
                        n = n[len('Frame'):]
                    enumerator_proto_name = n + 'EnumeratorArgs'
                    source_input.enumerator_args = python_to_proto(
                        self.protobufs, enumerator_proto_name, args)
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
                    sink_args = j.outputs.add()
                    sink_args.op_index = op_idx
                    # We special case on FrameColumn or Column sinks to catch
                    # the output table name
                    n = op._name
                    if n == 'FrameColumn' or n == 'Column':
                        assert isinstance(args, basestring)
                        output_table_name = args
                        job_output_table_names.append(args)
                    else:
                        # Encode the args
                        if n.startswith('Frame'):
                            n = n[len('Frame'):]
                        sink_proto_name = n + 'SinkStreamArgs'
                        sink_args.args = python_to_proto(
                            self.protobufs, sink_proto_name, args)
                        output_table_name = ''
                else:
                    raise ScannerException(
                        'Attempted to bind arguments to Op {} which is not '
                        'an input, sampling, spacing, slicing, or output Op.'
                        .format(
                            op.name()))  # FIXME(apoms): op.name() is unbound
            if is_table_output and output_table_name is None:
                raise ScannerException(
                    'Did not specify the output table name by binding a '
                    'string to the output Op.')
            j.output_table_name = output_table_name

        # Delete tables if they exist and force was specified
        if is_table_output:
            to_delete = []
            for name in job_output_table_names:
                if self.has_table(name):
                    if force:
                        to_delete.append(name)
                    else:
                        raise ScannerException(
                            'Job would overwrite existing table {}'.format(
                                name))
            self.delete_tables(to_delete)

        job_params.compression.extend(compression_options)
        job_params.pipeline_instances_per_node = (pipeline_instances_per_node
                                                  or -1)
        job_params.work_packet_size = work_packet_size
        job_params.io_packet_size = io_packet_size
        job_params.profiling = profiling
        job_params.tasks_in_queue_per_pu = tasks_in_queue_per_pu
        job_params.load_sparsity_threshold = load_sparsity_threshold
        job_params.boundary_condition = (
            self.protobufs.BulkJobParameters.REPEAT_EDGE)
        job_params.task_timeout = task_timeout
        job_params.checkpoint_frequency = checkpoint_frequency

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
        self._try_rpc(lambda: self._master.NewJob(
            job_params, timeout=self._grpc_timeout))

        job_status = self.wait_on_current_job(show_progress)

        if not job_status.result.success:
            raise ScannerException(job_status.result.msg)

        # Invalidate db metadata because of job run
        self._cached_db_metadata = None

        db_meta = self._load_db_metadata()
        job_id = None
        for job in db_meta.bulk_jobs:
            if job.name == job_name:
                job_id = job.id
        if job_id is None:
            raise ScannerException(
                'Internal error: job id not found after run')

        if is_table_output:
            return [self.table(t) for t in job_output_table_names]
        else:
            return []
