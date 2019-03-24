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
import cloudpickle
import types
from tqdm import tqdm
from typing import Sequence, List, Union, Tuple, Optional, Callable

if sys.platform == 'linux' or sys.platform == 'linux2':
    import prctl

from multiprocessing import Process, cpu_count, Event
import multiprocessing as mp
from subprocess import Popen, PIPE
from random import choice
from string import ascii_uppercase

from scannerpy.common import *
from scannerpy.profiler import Profiler
from scannerpy.config import Config
from scannerpy.op import OpGenerator, Op, OpColumn, SliceList
from scannerpy.source import SourceGenerator, Source
from scannerpy.sink import SinkGenerator, Sink
from scannerpy.streams import StreamsGenerator
from scannerpy.partitioner import TaskPartitioner
from scannerpy.table import Table
from scannerpy.column import Column
from scannerpy.protobufs import protobufs, python_to_proto
from scannerpy.job import Job
from scannerpy.kernel import Kernel
from scannerpy import types as scannertypes
from scannerpy.storage import StorageBackend, StoredStream, NamedVideoStream
from scannerpy.io import IOGenerator

import scannerpy._python as bindings
import scanner.metadata_pb2 as metadata_types
import scanner.engine.rpc_pb2 as rpc_types
import scanner.engine.rpc_pb2_grpc as grpc_types
import scanner.types_pb2 as misc_types



SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))



class Client(object):
    r"""Entrypoint for all Scanner operations.

    Parameters
    ----------
    master
      The address of the master process. The addresses should be formatted
      as 'ip:port'. If the `start_cluster` flag is specified, the Client
      object will ssh into the provided address and start a master process.
      You should have ssh access to the target machine and scannerpy should
      be installed.

    workers
      The list of addresses to spawn worker processes on. The addresses
      should be formatted as 'ip:port'. Like with `master`, you should have
      ssh access to the target machine and scannerpy should be installed. If
      `start_cluster` is false, this parameter has no effect.

    start_cluster
      If true, a master process and worker processes will be spawned at the
      addresses specified by `master` and `workers`, respectively.

    config_path
      Path to a Scanner configuration TOML, by default assumed to be
      '~/.scanner/config.toml'.

    config
      The scanner Config to use. If specified, config_path is ignored.

    debug
      This flag is only relevant when `start_cluster == True`. If true, the
      master and worker servers are spawned in the same process as the
      invoking python code, enabling easy gdb-based debugging.

    Other Parameters
    ----------------
    prefetch_table_metadata

    no_workers_timeout

    grpc_timeout


    Attributes
    ----------
    config : Config
      The Config object used to initialize this Client.

    ops : OpGenerator
      Represents the set of available Ops. Ops can be created like so:

      :code:`output = cl.ops.ExampleOp(arg='example')`

      For a more detailed description, see :class:`~scannerpy.op.OpGenerator`

    sources : SourceGenerator
      Represents the set of available Sources. Sources are created just like Ops.
      See :class:`~scannerpy.op.SourceGenerator`

    sinks : SinkGenerator
      Represents the set of available Sinks. Sinks are created just like Ops.
      See :class:`~scannerpy.op.SinkGenerator`

    streams : StreamsGenerator
      Used to specify which elements to sample from a sequence.
      See :class:`~scannerpy.streams.StreamsGenerator`

    partitioner : TaskPartitioner
      Used to specify how to split the elements in a sequence when performing a
      slice operation. See :class:`~scannerpy.partitioner.TaskPartitioner`.

    protobufs : ProtobufGenerator
      Used to construct protobuf objects that handle serialization/deserialization
      of the outputs of Ops.
    """

    def __init__(self,
                 master: str = None,
                 workers: List[str] = None,
                 start_cluster: bool = True,
                 config_path: str = None,
                 config: Config = None,
                 debug: bool = None,
                 enable_watchdog: bool = True,
                 prefetch_table_metadata: bool = True,
                 no_workers_timeout: float = 30,
                 grpc_timeout: float = 30,
                 new_job_retries_limit: int = 5,
                 machine_params = None):
        if config:
            self.config = config
        else:
            self.config = Config(config_path)

        self._start_cluster = start_cluster
        self._workers_started = False
        self._enable_watchdog = enable_watchdog
        # Worker always has watchdog enabled to determine when master
        # connection has failed
        self._enable_worker_watchdog = True
        self._prefetch_table_metadata = prefetch_table_metadata
        self._no_workers_timeout = no_workers_timeout
        self._debug = debug
        self._grpc_timeout = grpc_timeout
        self._new_job_retries_limit = new_job_retries_limit
        self._machine_params = machine_params
        if debug is None:
            self._debug = (master is None and workers is None)

        self._master = None

        self._bindings = bindings

        # Setup Client metadata
        self._db_path = self.config.db_path
        self._storage = self.config.storage
        self._cached_db_metadata = None
        self._png_dump_prefix = '__png_dump_{:s}_{:s}'

        self.ops = OpGenerator(self)
        self.sources = SourceGenerator(self)
        self.sinks = SinkGenerator(self)
        self.streams = StreamsGenerator(self)
        self.partitioner = TaskPartitioner(self)
        self.io = IOGenerator(self)
        self._op_cache = {}
        self._python_ops = set()
        self._modules = set()
        self._enumerator_info_cache = {}
        self._sink_info_cache = {}

        self._master_conn = None
        self._workers = {}
        self._worker_conns = None
        self._worker_paths = workers
        self.start_master(master)

    def __del__(self):
        # Client crashed during config creation if this attr is missing
        if hasattr(self, '_start_cluster') and self._start_cluster:
            self._stop_heartbeat()
            self.stop_cluster()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_val, exception_tb):
        self._stop_heartbeat()
        self.stop_cluster()
        del self._db

    def _load_descriptor(self, descriptor, path):
        d = descriptor()
        path = '{}/{}'.format(self._db_path, path)
        try:
            d.ParseFromString(self._storage.read(path))
        except UserWarning:
            raise ScannerException(
                'Internal error. Missing file {}'.format(path))
        return d

    def _save_descriptor(self, descriptor, path):
        self._storage.write(('{}/{}'.format(self._db_path, path)),
                            descriptor.SerializeToString())

    def _load_table_metadata(self, table_names):
        NUM_TABLES_TO_READ = 100000
        tables = []
        for i in range(0, len(table_names), NUM_TABLES_TO_READ):
            get_tables_params = protobufs.GetTablesParams()
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
            desc = self._load_descriptor(protobufs.DatabaseDescriptor,
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
                # Read all table descriptors from client
                table_names = list(self._table_name.keys())
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
        worker = protobufs.WorkerStub(channel)
        try:
            self._worker.Ping(
                protobufs.Empty(), timeout=self._grpc_timeout)
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
        self._master = protobufs.MasterStub(channel)
        result = False
        try:
            self._master.Ping(
                protobufs.Empty(), timeout=self._grpc_timeout)
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
        host_ip = socket.gethostbyname(host_name)
        if (ipaddress.ip_address(host_ip).is_loopback or
            host_name == 'localhost'):
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
        def heartbeat_task(stop_event, master_address, ppid):
            if sys.platform == 'linux' or sys.platform == 'linux2':
                prctl.set_pdeathsig(signal.SIGTERM)
            channel = self._make_grpc_channel(master_address)
            master = grpc_types.MasterStub(channel)
            while not stop_event.is_set():
                if os.getppid() != ppid:
                    return
                try:
                    master.PokeWatchdog(
                        rpc_types.Empty(), timeout=self._grpc_timeout)
                except grpc.RpcError as e:
                    pass
                time.sleep(1)

        self._heartbeat_stop_event = Event()
        pid = os.getpid()
        self._heartbeat_process = Process(
            target=heartbeat_task,
            args=(self._heartbeat_stop_event, self._master_address, pid))
        self._heartbeat_process.daemon = True
        self._heartbeat_process.start()

    def _stop_heartbeat(self):
        if (self._enable_watchdog and self._heartbeat_stop_event):
            self._heartbeat_stop_event.set()

    def _handle_signal(self, signum, frame):
        if (signum == signal.SIGINT or signum == signal.SIGTERM
                or signum == signal.SIGSEGV or signum == signal.SIGABRT):
            self._stop_heartbeat()
            self.stop_cluster()
            if signum == signal.SIGINT:
                sys.exit(0)
            else:
                sys.exit(1)

    def _try_rpc(self, fn):
        try:
            result = fn()
        except grpc.RpcError as e:
            raise ScannerException(e)

        if isinstance(result, protobufs.Result):
            if not result.success:
                raise ScannerException(result.msg)

        return result

    def _get_source_info(self, source_name):
        source_info_args = protobufs.SourceInfoArgs()
        source_info_args.source_name = source_name

        source_info = self._try_rpc(
            lambda: self._master.GetSourceInfo(source_info_args))

        if not source_info.result.success:
            raise ScannerException(source_info.result.msg)

        return source_info

    def _get_enumerator_info(self, enumerator_name):
        if enumerator_name in self._enumerator_info_cache:
            return self._enumerator_info_cache[enumerator_name]

        enumerator_info_args = protobufs.EnumeratorInfoArgs()
        enumerator_info_args.enumerator_name = enumerator_name

        enumerator_info = self._try_rpc(
            lambda: self._master.GetEnumeratorInfo(enumerator_info_args))

        if not enumerator_info.result.success:
            raise ScannerException(enumerator_info.result.msg)

        self._enumerator_info_cache[enumerator_name] = enumerator_info

        return enumerator_info

    def _get_sink_info(self, sink_name):
        if sink_name in self._sink_info_cache:
            return self._sink_info_cache[sink_name]

        sink_info_args = protobufs.SinkInfoArgs()
        sink_info_args.sink_name = sink_name

        sink_info = self._try_rpc(
            lambda: self._master.GetSinkInfo(sink_info_args))

        if not sink_info.result.success:
            raise ScannerException(sink_info.result.msg)

        self._sink_info_cache[sink_name] = sink_info

        return sink_info

    def _get_op_info(self, op_name):
        if op_name in self._op_cache:
            op_info = self._op_cache[op_name]
        else:
            op_info_args = protobufs.OpInfoArgs()
            op_info_args.op_name = op_name

            op_info = self._try_rpc(
                lambda: self._master.GetOpInfo(op_info_args, self._grpc_timeout)
            )

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

    def _toposort(self, outputs):
        # Perform DFS on modified graph
        edges = defaultdict(list)
        in_edges_left = defaultdict(int)

        source_nodes = []
        explored_nodes = set()
        stack = list(outputs)
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
        stream_ops = {}
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
                stream_ops[c] = op_idx
            elif isinstance(c, Sink):
                output_ops[c] = op_idx

        return eval_sorted, \
            eval_index, \
            source_ops, \
            stream_ops, \
            output_ops

    def _parse_size_string(self, s):
        (prefix, suffix) = (s[:-1], s[-1])
        mults = {'G': 1024**3, 'M': 1024**2, 'K': 1024**1}
        suffix = suffix.upper()
        if suffix not in mults:
            raise ScannerException('Invalid size suffix in "{}"'.format(s))
        return int(prefix) * mults[suffix]

    def load_op(self, so_path: str, proto_path: str = None):
        r"""Loads a custom op into the Scanner runtime.

        Parameters
        ----------
        so_path
          Path to the custom op's shared library (.so).

        proto_path
          Path to the custom op's arguments protobuf if one exists.

        Raises
        ------
        ScannerException
          Raised when the master fails to load the op.

        """
        if proto_path is not None:
            protobufs.add_module(proto_path)
        op_path = protobufs.OpPath()
        op_path.path = so_path
        self._try_rpc(
            lambda: self._master.LoadOp(op_path, timeout=self._grpc_timeout))
        self._modules.add((so_path, proto_path))

    def has_gpu(self):
        try:
            with open(os.devnull, 'w') as f:
                subprocess.check_call(['nvidia-smi'], stdout=f, stderr=f)
            return True
        except:
            pass
        return False

    def summarize(self) -> str:
        r"""Returns a human-readable summarization of the client state.
        """
        summary = ''
        db_meta = self._load_db_metadata()
        if len(db_meta.tables) == 0:
            return 'summarize: your client is empty!'

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

    def start_master(self, master: str):
        r"""Starts a Scanner master.

        Parameters
        ----------
        master
          ssh-able address of the master node.
        """

        if master is None:
            self._master_address = (
                self.config.master_address + ':' + self.config.master_port)
        else:
            self._master_address = master

        if ':' not in self._master_address:
            raise ScannerException(
                ('Did you forget to specify the master port number? '
                 'Specified address is {s:s}. It should look like {s:s}:5001')
                .format(s=self._master_address))

        # Start up heartbeat to keep master alive
        # NOTE(apoms): This MUST BE before any grpc channel is created, since it
        # forks a process and forking after channel creation causes hangs in the
        # forked process under grpc
        # https://github.com/grpc/grpc/issues/13873#issuecomment-358476408
        if self._enable_watchdog:
            self._start_heartbeat()

        # Boot up C++ database bindings
        self._db = self._bindings.Database(self.config.storage_config,
                                           str(self._db_path),
                                           str(self._master_address))

        if self._start_cluster:
            # Set handler to shutdown cluster on signal
            # TODO(apoms): we should clear these handlers when stopping
            # the cluster
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGSEGV, self._handle_signal)
            signal.signal(signal.SIGABRT, self._handle_signal)

            if self._debug:
                self._master_conn = None
                res = self._bindings.start_master(
                    self._db, self.config.master_port, SCRIPT_DIR, self._enable_watchdog,
                    self._no_workers_timeout, self._new_job_retries_limit).success
                assert res
                res = self._connect_to_master()
                if not res:
                    raise ScannerException(
                        'Failed to connect to local master process on port '
                        '{:s}. (Is there another process that is bound to that '
                        'port already?)'.format(self.config.master_port))

            else:
                master_port = self._master_address.partition(':')[2]
                # https://stackoverflow.com/questions/30469575/how-to-pickle-and-unpickle-to-portable-string-in-python-3
                pickled_config = pickle.dumps(self.config, 0).decode()
                master_cmd = (
                    'python3 -c ' + '\"from scannerpy import start_master\n' +
                    'import pickle\n' +
                    'config=pickle.loads(bytes(\'\'\'{config:s}\'\'\', \'utf8\'))\n'
                    + 'start_master(port=\'{master_port:s}\', block=True,\n' +
                    '             config=config, watchdog={watchdog},\n' +
                    '             no_workers_timeout={no_workers})\" ' +
                    '').format(
                        master_port=master_port,
                        config=pickled_config,
                        watchdog=self._enable_watchdog,
                        no_workers=self._no_workers_timeout)
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

    def start_workers(self, workers: List[str]):
        r"""Starts Scanner workers.

        Parameters
        ----------
        workers
          list of ssh-able addresses of the worker nodes.
        """
        if workers is None:
            self._worker_addresses = [
                self.config.master_address + ':' + self.config.worker_port
            ]
        else:
            self._worker_addresses = workers

        if self._debug:
            self._worker_conns = None
            machine_params = self._machine_params or self._bindings.default_machine_params()

            for i in range(len(self._worker_addresses)):
                start_worker(
                    self._master_address,
                    port=str(int(self.config.worker_port) + i),
                    config=self.config,
                    db=self._db,
                    watchdog=self._enable_worker_watchdog,
                    machine_params=machine_params)
        else:
            pickled_config = pickle.dumps(self.config, 0).decode()
            worker_cmd = (
                'python3 -c ' + '\"from scannerpy import start_worker\n' +
                'import pickle\n' +
                'config=pickle.loads(bytes(\'\'\'{config:s}\'\'\', \'utf8\'))\n'
                + 'start_worker(\'{master:s}\', port=\'{worker_port:s}\',\n' +
                '             block=True,\n' +
                '             watchdog={watchdog},' +
                '             config=config)\" ' + '')

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
                                watchdog=self._enable_worker_watchdog,
                                worker_port=w.partition(':')[2]),
                            nohup=True))
                except Exception as e:
                    print(
                        'WARNING: Failed to ssh into {:s} because: {:s}'.format(
                            w, repr(e)))
                    ignored_nodes += 1
            slept_so_far = 0
            # Has to be this long for GCS
            sleep_time = 60
            while slept_so_far < sleep_time:
                active_workers = self._master.ActiveWorkers(
                    protobufs.Empty(), timeout=self._grpc_timeout)
                if (len(active_workers.workers) > len(self._worker_conns)):
                    raise ScannerException(
                        ('Master has more workers than requested ' +
                         '({:d} vs {:d})').format(
                             len(active_workers.workers),
                             len(self._worker_conns)))
                if (len(active_workers.workers) == len(self._worker_conns)):
                    break
                time.sleep(0.3)
                slept_so_far += 0.3
            if slept_so_far >= sleep_time:
                self.stop_cluster()
                raise ScannerException(
                    'Timed out waiting for workers to connect to master')
            if ignored_nodes > 0:
                print(
                    'Ignored {:d} nodes during startup.'.format(ignored_nodes))

        self._workers_started = True

    def stop_cluster(self):
        r"""Stops the Scanner master and workers.
        """
        if self._start_cluster:
            if self._master:
                # Stop heartbeat
                self._stop_heartbeat()
                try:
                    self._try_rpc(
                        lambda: self._master.Shutdown(
                            protobufs.Empty(), timeout=self._grpc_timeout))
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

    def register_op(self,
                    name: str,
                    input_columns: List[Union[str, Tuple[str, ColumnType]]],
                    output_columns: List[Union[str, Tuple[str, ColumnType]]],
                    variadic_inputs: bool = False,
                    stencil: List[int] = None,
                    unbounded_state: bool = False,
                    bounded_state: int = None,
                    proto_path: str = None):
        r"""Register a new Op with the Scanner master.

        Parameters
        ----------
        name
          Name of the Op.

        input_columns
          A list of the inputs for this Op. Can be either the name of the input
          as a string or a tuple of ('name', ColumnType). If only the name is
          specified as a string, the ColumnType is assumed to be
          ColumnType.Blob.

        output_columns
          A list of the outputs for this Op. Can be either the name of the output
          as a string or a tuple of ('name', ColumnType). If only the name is
          specified as a string, the ColumnType is assumed to be
          ColumnType.Blob.

        variadic_inputs
          If true, this Op may take a variable number of inputs and
          `input_columns` is ignored. Variadic inputs are specified as
          positional arguments when invoking the Op, instead of keyword
          arguments.

        stencil
          Specifies the default stencil to use for the Op. If none, indicates
          that the the Op does not have the ability to stencil. A stencil of
          [0] should be specified if the Op can stencil but should not by
          default.

        unbounded_state
          If true, indicates that the Op needs to see all previous elements
          of its input sequences before it can compute a given element. For
          example, to compute output element at index 100, the Op must have
          already produced elements 0-99. This option is mutually exclusive
          with `bounded_state`.

        bounded_state
          If true, indicates that the Op needs to see all previous elements
          of its input sequences before it can compute a given element. For
          example, to compute output element at index 100, the Op must have
          already produced elements 0-99. This option is mutually exclusive
          with `bounded_state`.

        proto_path
          Optional path to the proto file that describes the configuration
          arguments to this Op.

        Raises
        ------
        ScannerException
          Raised when the master fails to register the Op.
        """
        op_registration = protobufs.OpRegistration()
        op_registration.name = name
        op_registration.variadic_inputs = variadic_inputs
        op_registration.has_unbounded_state = unbounded_state

        def add_col(columns, col):
            if isinstance(col, str):
                c = columns.add()
                c.name = col
                c.type = protobufs.Bytes
            elif isinstance(col, collections.Iterable):
                c = columns.add()
                c.name = col[0]
                c.type = ColumnType.to_proto(protobufs, col[1])
                c.type_name = col[2].cpp_name
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
            protobufs.add_module(proto_path)

        self._try_rpc(lambda: self._master.RegisterOp(
            op_registration, timeout=self._grpc_timeout))

    def register_python_kernel(
            self,
            op_name: str,
            device_type: DeviceType,
            kernel: Union[types.FunctionType, types.BuiltinFunctionType,
                          Kernel],
            batch: int = 1):
        r"""Register a Python Kernel with the Scanner master.

        Parameters
        ----------
        op_name
          Name of the Op.

        device_type
          The device type of the resource this kernel uses.

        kernel
          The class or function that implements the kernel.

        batch
          Specifies a default for how many elements this kernel should batch
          over. If `batch == 1`, kernel is assume to not be able to batch.

        Raises
        ------
        ScannerException
          Raised when the master fails to register the kernel.
        """

        if isinstance(kernel, types.FunctionType) or isinstance(
                kernel, types.BuiltinFunctionType):

            class KernelWrapper(Kernel):
                def __init__(self, config, **kwargs):
                    self._config = config

                def execute(self, columns):
                    return kernel(self._config, columns)

            kernel_cls = KernelWrapper
        else:
            kernel_cls = kernel

        py_registration = protobufs.PythonKernelRegistration()
        py_registration.op_name = op_name
        py_registration.device_type = DeviceType.to_proto(
            protobufs, device_type)
        py_registration.kernel_code = cloudpickle.dumps(kernel_cls)
        py_registration.batch_size = batch
        self._try_rpc(
            lambda: self._master.RegisterPythonKernel(
                py_registration, timeout=self._grpc_timeout))
        self._python_ops.add(op_name)

    def ingest_videos(
            self,
            videos: List[Tuple[str, str]],
            inplace: bool = False,
            force: bool = False) -> Tuple[List[Table], List[Tuple[str, str]]]:
        r"""Creates tables from videos.

        Parameters
        ----------
        videos
          The list of videos to ingest into the client. Each element in the
          list should be ('table_name', 'path/to/video').

        inplace
          If true, ingests the videos without copying them into the client.
          Currently only supported for mp4 containers.

        force
          If true, deletes existing tables with the same names.

        Returns
        -------
        tables: List[Table]
          List of table objects for the ingested videos.

        failures: List[Tuple[str, str]]
          List of ('path/to/video', 'reason for failure') tuples for each video
          which failed to ingest.
        """

        if len(videos) == 0:
            raise ScannerException('Must ingest at least one video.')

        [table_names, paths] = list(zip(*videos))
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
        ingest_params = protobufs.IngestParameters()
        ingest_params.table_names.extend(table_names)
        ingest_params.video_paths.extend(paths)
        ingest_params.inplace = inplace
        ingest_result = self._try_rpc(
            lambda: self._master.IngestVideos(ingest_params))
        if not ingest_result.result.success:
            raise ScannerException(ingest_result.result.msg)
        failures = list(
            zip(ingest_result.failed_paths, ingest_result.failed_messages))

        self._cached_db_metadata = None
        return ([
            self.table(t) for (t, p) in videos
            if p not in ingest_result.failed_paths
        ], failures)

    def has_table(self, name: str) -> bool:
        r"""Checks if a table exists in the database.

        Parameters
        ----------
        name
          The name of the table to check for.

        Returns
        -------
        bool
          True if the table exists, false otherwise.
        """
        db_meta = self._load_db_metadata()
        if name in self._table_name:
            return True
        return False

    def delete_tables(self, names: List[str]):
        r"""Deletes tables from the database.

        Parameters
        ----------
        names
          The names of the tables to delete.
        """
        delete_tables_params = protobufs.DeleteTablesParams()
        for name in names:
            delete_tables_params.tables.append(name)
        self._try_rpc(lambda: self._master.DeleteTables(delete_tables_params))
        self._cached_db_metadata = None

    def delete_table(self, name: str):
        r"""Deletes a table from the database.

        Parameters
        ----------
        name
          The name of the table to delete.
        """
        self.delete_tables([name])

    def new_table(self,
                  name: str,
                  columns: List[str],
                  rows: List[List[bytes]],
                  fns=None,
                  force: bool = False) -> Table:
        r"""Creates a new table from a list of rows.

        Parameters
        ----------
        name
          String name of the table to create

        columns
          List of names of table columns

        rows
          List of rows with each row a list of elements corresponding
          to the specified columns. Elements must be strings of
          serialized representations of the data.

        fns

        force

        Returns
        -------
        Table
          The new table object.
        """

        if self.has_table(name):
            if force:
                self.delete_table(name)
            else:
                raise ScannerException(
                    'Attempted to create table with existing '
                    'name {}'.format(name))
        if fns is not None:
            rows = [[fn(col, protobufs) for fn, col in zip(fns, row)]
                    for row in rows]

        params = protobufs.NewTableParams()
        params.table_name = name
        params.columns[:] = columns

        for i, row in enumerate(rows):
            row_proto = params.rows.add()
            row_proto.columns[:] = row

        self._try_rpc(lambda: self._master.NewTable(params))

        self._cached_db_metadata = None

        return self.table(name)

    def table(self, name: str) -> Table:
        r"""Retrieves a Table.

        Parameters
        ----------
        name
          Name of the table to retrieve.

        Returns
        -------
        Table
          The table object.
        """
        db_meta = self._load_db_metadata()

        table_name = None
        table_id = None
        if isinstance(name, str):
            table_name = name
            if name in self._table_name:
                table_id = db_meta.tables[self._table_name[name]].id
        elif isinstance(name, int):
            table_id = name
            if name in self._table_id:
                table_name = db_meta.tables[self._table_id[name]].name
        else:
            raise ScannerException('Invalid table identifier')

        table = Table(self, table_name, table_id)
        if self._prefetch_table_metadata and table_id in self._table_descriptor:
            table._descriptor = self._table_descriptor[table_id]

        return table

    def sequence(self, name: str) -> Column:
        t = self.table(name)
        if t.committed():
            column_names = t.column_names()
            return t.column('frame' if 'frame' in column_names else 'column')
        else:
            return t.column('column')

    def profiler(self, job_name, **kwargs):
        db_meta = self._load_db_metadata()
        if isinstance(job_name, str):
            job_id = None
            for job in db_meta.bulk_jobs:
                if job.name == job_name:
                    job_id = job.id
                    break
            if job_id is None:
                raise ScannerException(
                    'Job name {} does not exist'.format(job_name))
        else:
            job_id = job_name

        return Profiler(self, job_id, **kwargs)

    def get_active_jobs(self):
        req = protobufs.GetJobsRequest()
        reply = self._try_rpc(lambda: self._master.GetJobs(
            req, timeout=self._grpc_timeout))
        return [x for x in reply.active_bulk_jobs]

    def wait_on_job_gen(self, bulk_job_id, show_progress=True):
        pbar = None
        total_tasks = None
        last_task_count = 0
        last_jobs_failed = 0
        last_failed_workers = 0
        while True:
            try:
                req = protobufs.GetJobStatusRequest()
                req.bulk_job_id = bulk_job_id
                job_status = self._master.GetJobStatus(
                    req, timeout=self._grpc_timeout)
                if show_progress and pbar is None and job_status.total_jobs != 0 \
                   and job_status.total_tasks != 0:
                    total_tasks = job_status.total_tasks
                    # Lower smoothing provides more accurate ETAs over long jobs.
                    # See: https://tqdm.github.io/docs/tqdm/
                    pbar = tqdm(total=total_tasks, smoothing=0.01)
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
            yield

        if pbar is not None:
            pbar.update(total_tasks - last_task_count)
            pbar.close()

        yield job_status
        return

    def wait_on_job(self, *args, **kwargs):
        sleep_schedule = []
        for i in range(15):
            sleep_schedule += [0.02 * (2 ** i)] * (50 // (2 ** i))
        gen = self.wait_on_job_gen(*args, **kwargs)
        sleep_attempt = 0
        while True:
            try:
                job_status = next(gen)
                if sleep_attempt < len(sleep_schedule):
                    time.sleep(sleep_schedule[sleep_attempt])
                else:
                    time.sleep(1.0)
                sleep_attempt += 1
            except StopIteration:
                break
        return job_status

    def bulk_fetch_video_metadata(self, tables):
        params = protobufs.GetVideoMetadataParams(
            tables=[t.name() for t in tables])
        result = self._try_rpc(lambda: self._master.GetVideoMetadata(
            params, timeout=self._grpc_timeout))
        return result.videos

    def batch_load(self, tables, column, callback, batch_size=1000, workers=16):
        def batch(l, n):
            for i in range(0, len(l), n):
                yield l[i:i+n]

        with mp.get_context('spawn').Pool(processes=workers) as pool:
            for _ in tqdm(pool.imap_unordered(
                    _batch_load_column,
                    [(l, column, callback) for l in
                     batch([t.name() for t in tables], batch_size)])):
                pass

    def run(self,
            outputs: Union[Sink, List[Sink]],
            perf_params: Callable[[], PerfParams],
            cache_mode: CacheMode = CacheMode.Error,
            cpu_pool: str = None,
            gpu_pool: str = None,
            pipeline_instances_per_node: int = None,
            show_progress: bool = True,
            profiling: bool = False,
            load_sparsity_threshold: int = 8,
            tasks_in_queue_per_pu: int = 4,
            task_timeout: int = 0,
            checkpoint_frequency: int = 10,
            detach: bool = False,
            profiler_level: int = 1):
        r"""Runs a collection of jobs.

        Parameters
        ----------
        outputs
          The Sink or Sinks that should be processed.

        cache_mode
          Determines whether to overwrite, ignore, or raise an error when running a job on
          existing outputs.

        cpu_pool

        gpu_pool

        pipeline_instances_per_node
          The number of concurrent instances of the computation graph to
          execute. If set to None, it will be automatically inferred based on
          computation graph and the available machine resources.

        show_progress
          If true, will display an ASCII progress bar measuring job status.

        profiling

        Other Parameters
        ----------------
        load_sparsity_threshold

        tasks_in_queue_per_pu

        task_timeout

        checkpoint_frequency

        Returns
        -------
        int
          The job id.
        """

        if not isinstance(outputs, list):
            outputs = [outputs]

        sorted_ops, op_index, source_ops, stream_ops, output_ops = (
            self._toposort(outputs))

        for op in source_ops.keys():
            streams = op._outputs[0]._streams
            if isinstance(streams[0], NamedVideoStream):
                streams[0].storage().ingest(self, streams)

        # Collect compression annotations to add to job
        output_column_names = []
        compression_options = []
        for op in sorted_ops:
            if op in outputs:
                for out_col in op.inputs():
                    opts = protobufs.OutputColumnCompression()
                    opts.codec = 'default'
                    if out_col._type == protobufs.Video:
                        for k, v in out_col._encode_options.items():
                            if k == 'codec':
                                opts.codec = v
                            else:
                                opts.options[k] = str(v)
                    compression_options.append(opts)
                # Get output columns
                output_column_names += op._output_names

        job_params = protobufs.BulkJobParameters()
        job_name = ''.join(choice(ascii_uppercase) for _ in range(12))
        job_params.job_name = job_name
        job_params.ops.extend([e.to_proto(op_index) for e in sorted_ops])
        job_params.output_column_names.extend(output_column_names)

        N = None
        for op in sorted_ops:
            n = None
            if op._job_args is not None:
                n = len(op._job_args)
            elif op._extra is not None and 'job_args' in op._extra:
                if not isinstance(op._extra['job_args'], list):
                    raise ScannerException("Per-stream arguments to op `{}` must be a list." \
                                           .format(op._name))
                n = len(op._extra['job_args'])
            else:
                continue

            if N is None:
                N = n
            elif n != N:
                raise ScannerException("Op `{}` had {} per-stream arguments, but expected {}" \
                                       .format(op._name, n, N))
        assert N is not None

        # Collect set of existing stored streams for each output
        output_ops_list = list(output_ops.keys())
        to_delete = []
        for op in output_ops_list:
            streams = op._streams
            storage = streams[0].storage()
            to_delete.append(set([i for i, s in enumerate(streams) if s.exists()]))

        # Check that all outputs have matching existing stored streams
        for i, s in enumerate(to_delete):
            for j, t in enumerate(to_delete):
                if i == j: continue
                diff = s - t
                if len(diff) > 0 :
                    raise Exception(
                        ("Output for stream {} exists in {} but does not exist in {}. Either both or"
                         "neither should exist.").format(
                            next(iter(diff)), output_ops_list[i]._name, output_ops_list[j]._name))

        to_cache = []
        if len(to_delete[0]) > 0:
            if cache_mode == CacheMode.Error:
                raise ScannerException(
                    ("Running this job would overwrite output `{}` of op `{}`. You can "
                     "change this behavior using sc.run(cache_mode=CacheMode.Ignore) to "
                     "ignore outputs that already exist, or CacheMode.Overwrite to "
                     "overwrite them.").format(next(iter(to_delete[0])), output_ops_list[0]._name))

            elif cache_mode == CacheMode.Overwrite:
                for op, td in zip(output_ops_list, to_delete):
                    streams = op._streams
                    storage = streams[0].storage()
                    storage.delete(self, [streams[i] for i in td])

            elif cache_mode == CacheMode.Ignore:
                to_cache = to_delete[0]

        # Struct of arrays to array of structs conversion
        jobs = []
        for i in range(N):
            if i in to_cache:
                continue

            op_args = {}
            for op in sorted_ops:
                if op._job_args is not None:
                    op_args[op] = op._job_args[i]
                elif op._extra is not None and 'job_args' in op._extra:
                    op_args[op] = op._extra['job_args'][i]
            jobs.append(Job(op_args=op_args))

        N -= len(to_cache)

        if len(jobs) == 0:
            return None

        for job in jobs:
            j = job_params.jobs.add()
            op_to_op_args = job.op_args()

            for op in sorted_ops:
                if op in source_ops:
                    op_idx = source_ops[op]
                    if not op in op_to_op_args:
                        raise ScannerException(
                            'No arguments bound to source {:s}.'.format(
                                op._name))
                    else:
                        args = op_to_op_args[op]

                    args = op_to_op_args[op]
                    source_input = j.inputs.add()
                    source_input.op_index = op_idx

                    source_input.enumerator_args = args

                elif op in stream_ops:
                    op_idx = stream_ops[op]
                    # If this is an Unslice Op, ignore it since it has no args
                    if op._name == 'Unslice':
                        continue
                    args = op_to_op_args[op]

                    saa = j.sampling_args_assignment.add()
                    saa.op_index = op_idx

                    if not isinstance(args, SliceList):
                        args = SliceList([args])

                    arg_builder = op._extra['arg_builder']
                    for arg in args:
                        # Construct the sampling_args using the arg_builder
                        if isinstance(arg, tuple):
                            # Positional arguments
                            sargs = arg_builder(*arg)
                        elif isinstance(arg, dict):
                            # Keyword arguments
                            sargs = arg_builder(**arg)
                        else:
                            # Single argument
                            sargs = arg_builder(arg)

                        sa = saa.sampling_args.add()
                        sa.CopyFrom(sargs)

                elif op in output_ops:
                    op_idx = output_ops[op]
                    sink_args = j.outputs.add()
                    sink_args.op_index = op_idx

                    args = op_to_op_args[op]

                    sink_args.args = args

                else:
                    # Regular old Op
                    op_idx = op_index[op]
                    oargs = j.op_args.add()
                    oargs.op_index = op_idx

                    if not op in op_to_op_args:
                        continue
                    else:
                        args = op_to_op_args[op]

                    if not isinstance(args, SliceList):
                        args = SliceList([args])

                    for arg in args:
                        oargs.op_args.append(arg)
                        #oargs.op_args.append(serialize_args(arg))

        perf_params = perf_params(
            inputs=[op._outputs[0]._streams for op in source_ops.keys()],
            ops=sorted_ops,
            tasks_in_queue_per_pu=tasks_in_queue_per_pu)

        job_params.compression.extend(compression_options)

        job_params.pipeline_instances_per_node = (pipeline_instances_per_node or -1)
        job_params.work_packet_size = perf_params.work_packet_size
        job_params.io_packet_size = perf_params.io_packet_size
        job_params.profiling = profiling
        job_params.tasks_in_queue_per_pu = tasks_in_queue_per_pu
        job_params.load_sparsity_threshold = load_sparsity_threshold
        job_params.boundary_condition = (
            protobufs.BulkJobParameters.REPEAT_EDGE)
        job_params.task_timeout = task_timeout
        job_params.checkpoint_frequency = checkpoint_frequency
        job_params.profiler_level = profiler_level

        job_params.memory_pool_config.pinned_cpu = False
        if cpu_pool is not None:
            job_params.memory_pool_config.cpu.use_pool = True
            if cpu_pool[0] == 'p':
                job_params.memory_pool_config.pinned_cpu = True
                cpu_pool = cpu_pool[1:]
            size = self._parse_size_string(cpu_pool)
            job_params.memory_pool_config.cpu.free_space = size
        else:
            job_params.memory_pool_config.cpu.use_pool = False

        if gpu_pool is not None:
            job_params.memory_pool_config.gpu.use_pool = True
            size = self._parse_size_string(gpu_pool)
            job_params.memory_pool_config.gpu.free_space = size
        else:
            job_params.memory_pool_config.gpu.use_pool = False

        if not self._workers_started and self._start_cluster:
            self.start_workers(self._worker_paths)

        # Invalidate db metadata because of job run
        self._cached_db_metadata = None

        # Run the job
        result = self._try_rpc(lambda: self._master.NewJob(
            job_params, timeout=self._grpc_timeout))

        bulk_job_id = result.bulk_job_id

        if detach:
            return bulk_job_id

        job_status = self.wait_on_job(bulk_job_id, show_progress)

        if not job_status.result.success:
            raise ScannerException(job_status.result.msg)

        db_meta = self._load_db_metadata()

        return bulk_job_id


def start_master(port: int = None,
                 config: Config = None,
                 config_path: str = None,
                 block: bool = False,
                 watchdog: bool = True,
                 no_workers_timeout: float = 30,
                 new_job_retries_limit: int = 5):
    r""" Start a master server instance on this node.

    Parameters
    ----------
    port
      The port number to start the master on. If unspecified, it will be
      read from the provided Config.

    config
      The scanner Config to use. If specified, config_path is ignored.

    config_path : optional
      Path to a Scanner configuration TOML, by default assumed to be
      `~/.scanner/config.toml`.

    block : optional
      If true, will wait until the server is shutdown. Server will not
      shutdown currently unless wait_for_server_shutdown is eventually
      called.

    watchdog : optional
      If true, the master will shutdown after a time interval if
      PokeWatchdog is not called.

    no_workers_timeout : optional
      The interval after which the master will consider a job to have failed if
      it has no workers connected to it.

    Returns
    -------
    Database
      A cpp database instance.
    """

    config = config or Config(config_path)
    port = port or config.master_port

    # Load all protobuf types
    db = bindings.Database(config.storage_config, config.db_path,
                           (config.master_address + ':' + port))
    result = bindings.start_master(db, port, SCRIPT_DIR, watchdog,
                                   no_workers_timeout,
                                   new_job_retries_limit)
    if not result.success():
        raise ScannerException('Failed to start master: {}'.format(
            result.msg()))
    if block:
        bindings.wait_for_server_shutdown(db)
    return db


def start_worker(master_address: str,
                 machine_params=None,
                 port: int = None,
                 config: Config = None,
                 config_path: str = None,
                 block: bool = False,
                 watchdog: bool = True,
                 db: Client = None):
    r"""Starts a worker instance on this node.

    Parameters
    ----------
    master_address
      The address of the master server to connect this worker to. The expected
      format is '0.0.0.0:5000' (ip:port).

    machine_params
      Describes the resources of the machine that the worker should manage. If
      left unspecified, the machine resources will be inferred.

    config
      The Config object to use in creating the worker. If specified, config_path
      is ignored.

    config_path
      Path to a Scanner configuration TOML, by default assumed to be
      `~/.scanner/config.toml`.

    block
      If true, will wait until the server is shutdown. Server will not shutdown
      currently unless wait_for_server_shutdown is eventually called.

    watchdog
      If true, the worker will shutdown after a time interval if
      PokeWatchdog is not called.

    Other Parameters
    ----------------
    db
      This is for internal usage only.

    Returns
    -------
    Database
      A cpp database instance.
    """

    # Worker always has watchdog enabled to determine when master connection
    # has failed
    if not watchdog:
        print(('Forcing worker to enable watchdog. Watchdog must be enabled to '
               'detect if worker has disconnected from master.'),
              file=sys.stderr)
        watchdog = True

    config = config or Config(config_path)
    port = port or config.worker_port

    # Load all protobuf types
    db = db or bindings.Database(
        config.storage_config,
        config.db_path,
        master_address)
    machine_params = machine_params or bindings.default_machine_params()
    result = bindings.start_worker(db, machine_params, str(port), SCRIPT_DIR,
                                   watchdog)

    if not result.success():
        raise ScannerException('Failed to start worker: {}'.format(
            result.msg()))

    if block:
        bindings.wait_for_server_shutdown(db)

    return result


def _batch_load_column(arg):
    (tables, column, callback) = arg
    sc = Client(start_cluster=False, enable_watchdog=False)
    for t in tables:
        callback(t, list(sc.table(t).column(column).load(workers=1)))
