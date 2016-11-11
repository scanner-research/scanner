from enum import Enum
from collections import defaultdict
import struct
import sys
import toml
import os
import logging
import sys
import tempfile
import time
import subprocess
import re

is_py3 = sys.version_info.major == 3
maxint = sys.maxsize if is_py3 else sys.maxint

DEVNULL = open(os.devnull, 'wb', 0)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def read_advance(fmt, buf, offset):
    new_offset = offset + struct.calcsize(fmt)
    return struct.unpack_from(fmt, buf, offset), new_offset


def unpack_string(buf, offset):
    s = ''
    while True:
        t, offset = read_advance('B', buf, offset)
        c = t[0]
        if c == 0:
            break
        s += str(chr(c))
    return s, offset


def parse_profiler_output(bytes_buffer, offset):
    # Node
    t, offset = read_advance('q', bytes_buffer, offset)
    node = t[0]
    # Worker type name
    worker_type, offset = unpack_string(bytes_buffer, offset)
    # Worker tag
    worker_tag, offset = unpack_string(bytes_buffer, offset)
    # Worker number
    t, offset = read_advance('q', bytes_buffer, offset)
    worker_num = t[0]
    # Number of keys
    t, offset = read_advance('q', bytes_buffer, offset)
    num_keys = t[0]
    # Key dictionary encoding
    key_dictionary = {}
    for i in range(num_keys):
        key_name, offset = unpack_string(bytes_buffer, offset)
        t, offset = read_advance('B', bytes_buffer, offset)
        key_index = t[0]
        key_dictionary[key_index] = key_name
    # Intervals
    t, offset = read_advance('q', bytes_buffer, offset)
    num_intervals = t[0]
    intervals = []
    for i in range(num_intervals):
        # Key index
        t, offset = read_advance('B', bytes_buffer, offset)
        key_index = t[0]
        t, offset = read_advance('q', bytes_buffer, offset)
        start = t[0]
        t, offset = read_advance('q', bytes_buffer, offset)
        end = t[0]
        intervals.append((key_dictionary[key_index], start, end))
    # Counters
    t, offset = read_advance('q', bytes_buffer, offset)
    num_counters = t[0]
    counters = {}
    for i in range(num_counters):
        # Counter name
        counter_name, offset = unpack_string(bytes_buffer, offset)
        # Counter value
        t, offset = read_advance('q', bytes_buffer, offset)
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


class ScannerConfig(object):
    """ TODO(wcrichto): document me """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self.default_config_path()
        config = self.load_config(config_path)
        try:
            self.scanner_path = config['scanner_path']
            sys.path.append('{}/build'.format(self.scanner_path))
            sys.path.append('{}/thirdparty/build/bin/storehouse/lib'
                            .format(self.scanner_path))

            from storehousepy import StorageConfig, StorageBackend
            storage = config['storage']
            storage_type = storage['type']
            self.db_path = storage['db_path']
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
        except KeyError as key:
            logging.critical('Scanner config missing key: {}'.format(key))
            exit()
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


class Sampling(Enum):
    All = 0
    Strided = 1
    Gather = 2
    SequenceGather = 3


class JobLoadException(Exception):
    pass


class JobResult(object):
    """ TODO(apoms): document me """

    def __init__(self, scanner, dataset_name, job_name, column, load_fn):
        self._scanner = scanner
        self._dataset_name = dataset_name
        self._job_name = job_name
        self._column = column
        self._load_fn = load_fn
        self._storage = self._scanner.config.storage
        self._db_path = self._scanner.config.db_path

        self._dataset = self._scanner._meta.DatasetDescriptor()
        self._dataset.ParseFromString(
            self._storage.read('{}/datasets/{}/descriptor.bin'
                               .format(self._db_path, dataset_name)))

        self._job = self._scanner._meta.JobDescriptor()
        self._job.ParseFromString(
            self._storage.read(
                '{}/datasets/{}/jobs/{}/descriptor.bin'
                .format(self._db_path, dataset_name, job_name)))

    def _load_output_file(self, video, video_name, work_item_index, rows,
                          istart, iend):
        try:
            contents = self._storage.read(
                '{}/datasets/{}/jobs/{}/{}_{}_{}.bin'.format(
                    self._db_path, self._dataset_name, self._job_name,
                    video_name, self._column, work_item_index))
            lens = []
            start_pos = maxint
            pos = 0
            num_rows = struct.unpack("l", contents[:8])
            assert num_rows > max(rows)

            i = 8
            for fi in rows:
                (buf_len,) = struct.unpack("l", contents[i:i+8])
                i += 8
                old_pos = pos
                pos += buf_len
                if (fi >= istart and fi <= iend):
                    if start_pos == maxint:
                        start_pos = old_pos
                    lens.append(buf_len)

            bufs = []
            i = 8 + len(rows) * 8 + start_pos
            for buf_len in lens:
                buf = contents[i:i+buf_len]
                i += buf_len
                item = self._load_fn(buf, video)
                bufs.append(item)

            return bufs
        except UserWarning:
            raise JobLoadException('Column {} does not exist for job {}'.format(
                self._column, self._job_name))

    def _load_video_descriptor(self, video_name):
        video = self._scanner._meta.VideoDescriptor()
        video.ParseFromString(
            self._storage.read(
                '{}/datasets/{}/data/{}_metadata.bin'.format(
                    self._db_path, self._dataset_name, video_name)))
        return video

    def _load_all_sampling(self, interval=None):
        item_size = self._job.work_item_size
        work_item_index = 0
        for vi, video_name in enumerate(self._dataset.video_names):
            video = self._load_video_descriptor(video_name)

            intervals = [i for i in range(0, video.frames - 1, item_size)]
            intervals.append(video.frames)
            intervals = zip(intervals[:-1], intervals[1:])
            assert(intervals is not None)

            result = {'video': video_name,
                      'frames': [],
                      'buffers': []}
            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)

            for i, ivl in enumerate(intervals):
                start = ivl[0]
                end = ivl[1]
                if start > iend or end < istart: continue
                result['buffers'] += self._load_output_file(video,
                                                            video_name,
                                                            work_item_index,
                                                            range(start, end),
                                                            istart,
                                                            iend)
                result['frames'] += range(start, end)
                work_item_index += 1
            yield result

    def _load_stride_sampling(self, interval=None):
        item_size = self._job.work_item_size
        stride = self._job.stride
        work_item_index = 0
        for vi, video_name in enumerate(self._dataset.video_names):
            video = self._load_video_descriptor(video_name)

            intervals = [i for i in range(0, video.frames - 1,
                                          item_size * stride)]
            intervals.append(video.frames)
            intervals = zip(intervals[:-1], intervals[1:])
            assert(intervals is not None)

            result = {'video': video_name,
                      'frames': [],
                      'buffers': []}
            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for i, ivl in enumerate(intervals):
                start = ivl[0]
                end = ivl[1]
                if start > iend or end < istart: continue
                rows = (end - start) / stride
                rows = range(start, end, stride)
                result['buffers'] += self._load_output_file(video,
                                                            video_name,
                                                            work_item_index,
                                                            rows,
                                                            istart,
                                                            iend)
                result['frames'] += range(start, end, stride)
                work_item_index += 1
            yield result

    def _load_gather_sampling(self, interval=None):
        item_size = self._job.work_item_size
        work_item_index = 0
        for samples in self._job.gather_points:
            video_index = samples.video_index
            video_name = self._dataset.video_names[video_index]
            video = self._load_video_descriptor(video_name)

            work_items = chunks(samples.frames, item_size)
            assert(work_items is not None)

            result = {'video': video_name,
                      'frames': [],
                      'buffers': []}
            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for i, item in enumerate(work_items):
                start = item[0]
                end = item[-1]
                if start > iend or end < istart: continue
                rows = item
                result['buffers'] += self._load_output_file(video,
                                                            video_name,
                                                            work_item_index,
                                                            rows,
                                                            istart,
                                                            iend)
                result['frames'] += item
                work_item_index += 1
            yield result

    def _load_sequence_gather_sampling(self, interval=None):
        item_size = self._job.work_item_size
        work_item_index = 0
        for samples in self._job.gather_sequences:
            video_index = samples.video_index
            video_name = self._dataset.video_names[video_index]
            video = self._load_video_descriptor(video_name)

            sequences = samples.intervals

            result = {'video': video_name,
                      'sequences': [(s.start, s.end) for s in sequences],
                      'frames': [],
                      'buffers': []}

            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for intvl in sequences:
                intervals = [i for i in range(intvl.start, intvl.end - 1,
                                              item_size)]
                intervals.append(intvl.end)
                intervals = zip(intervals[:-1], intervals[1:])
                assert(intervals is not None)

                for i, intvl in enumerate(intervals):
                    start = intvl[0]
                    end = intvl[1]
                    if start > iend or end < istart: continue
                    rows = range(start, end)
                    result['buffers'] += self._load_output_file(video,
                                                                video_name,
                                                                work_item_index,
                                                                rows,
                                                                istart,
                                                                iend)
                    result['frames'] += range(start, end)
                    work_item_index += 1
            yield result

    def get_sampling_type(self):
        JD = self._scanner._meta.JobDescriptor
        js = self._job.sampling
        s = None
        if js == JD.All:
            s = Sampling.All
        elif js == JD.Strided:
            s = Sampling.Strided
        elif js == JD.Gather:
            s = Sampling.Gather
        elif js == JD.SequenceGather:
            s = Sampling.SequenceGather
        return s

    def as_outputs(self, interval=None):
        sampling = self.get_sampling_type()
        if sampling is Sampling.All:
            return self._load_all_sampling(interval)
        elif sampling is Sampling.Strided:
            return self._load_stride_sampling(interval)
        elif sampling is Sampling.Gather:
            return self._load_gather_sampling(interval)
        elif sampling is Sampling.SequenceGather:
            return self._load_sequence_gather_sampling(interval)

    def as_frame_list(self, interval=None):
        for d in self.as_outputs(interval):
            yield (d['video'], zip(d['frames'], d['buffers']))

    def as_sequences(self, interval=None):
        """ TODO(apoms): implement """
        if self.get_sampling_type() != Sampling.SequenceGather:
            logging.error("")
            return
        pass

class Scanner(object):
    """ TODO(wcrichto): document me """

    def __init__(self, config_path=None):
        self.config = ScannerConfig(config_path)
        sys.path.append('{}/build'.format(self.config.scanner_path))
        sys.path.append('{}/thirdparty/build/bin/storehouse/lib'
                        .format(self.config.scanner_path))
        from scannerpy import metadata_pb2
        self._meta = metadata_pb2
        self._executable_path = (
            '{}/build/scanner_server'.format(self.config.scanner_path))
        self._storage = self.config.storage
        self._db_path = self.config.db_path

    def _load_descriptor(self, descriptor, path):
        d = descriptor()
        d.ParseFromString(self._storage.read(path))
        return d

    def get_job_result(self, dataset_name, job_name, column, fn):
        return JobResult(self, dataset_name, job_name, column, fn)

    def write_output_buffers(dataset_name, job_name, ident, column, fn, video_data):
        logging.critical('write_output_buffers is out of date. Needs fixing')
        exit()

        job = self._meta.JobDescriptor()
        job.id = ident

        col = job.columns.add()
        col.id = 0
        col.name = column

        for i, data in enumerate(video_data):
            start = 0
            end = len(data)
            video = job.videos.add()
            video.index = i
            interval = video.intervals.add()
            interval.start = start
            interval.end = end

            path = '{}/datasets/{}/jobs/{}/{}_{}_{}-{}.bin'.format(
                self.config.db_path, dataset_name, job_name, str(i), column,
                start, end)

            with open(path, 'wb') as f:
                all_bytes = ""
                for d in data:
                    byts = fn(d)
                    all_bytes += byts
                    f.write(struct.pack("=Q", len(byts)))
                f.write(all_bytes)

        with open('{}/datasets/{}/jobs/{}/descriptor.bin'
                  .format(self.config.db_path, dataset_name, job_name), 'wb') as f:
            f.write(job.SerializeToString())

    def loader(self, column):
        def decorator(f):
            def loader(dataset_name, job_name):
                return self.get_job_result(dataset_name, job_name, column, f)
            return loader
        return decorator

    def load_db_metadata(self):
        return self._load_descriptor(
            self._meta.DatabaseDescriptor,
            '{}/db_metadata.bin'.format(self._db_path))

    def write_db_metadata(self, meta):
        self._storage.write('{}/db_metadata.bin'.format(self._db_path),
                            meta.SerializeToString())

    def ingest(self, dataset_name, video_paths, opts={}):
        def gopt(k, default):
            return opts[k] if k in opts else default
        force = gopt('force', False)

        # Write video paths to temp file
        fd, paths_file_name = tempfile.mkstemp()
        with open(paths_file_name, 'w') as f:
            for p in video_paths:
                f.write(p + '\n')
            # Erase last newline
            f.seek(-1, 1)
            f.flush()

        current_env = os.environ.copy()
        start = time.time()
        cmd = ['mpirun',
               '-n', str(1),
               '--bind-to', 'none',
               self._executable_path,
               'ingest', dataset_name, paths_file_name]

        if force:
            cmd.append('-f')

        p = subprocess.Popen(
            cmd, env=current_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        success = (rc == 0)
        if not success:
            print(so)
        return success, elapsed

    @staticmethod
    def base_job_name():
        return 'base'

    def run(self, dataset_name, in_job_name, pipeline_name, out_job_name,
            opts={}):
        def gopt(k, default):
            return opts[k] if k in opts else default
        force = gopt('force', False)
        node_count = gopt('node_count', 1)
        pus_per_node = gopt('pus_per_node', 1)
        work_item_size = gopt('work_item_size', None)
        tasks_in_queue_per_pu = gopt('tasks_in_queue_per_pu', None)
        load_workers_per_node = gopt('tasks_in_queue_per_pu', None)
        save_workers_per_node = gopt('save_workers_per_node', None)
        custom_env = gopt('env', None)

        cmd = [['mpirun',
                '-n', str(node_count),
                '--bind-to', 'none',
                self._executable_path,
                'run', dataset_name, in_job_name, pipeline_name, out_job_name]]

        def add_opt(name, opt):
            if opt:
                cmd[0] += ['--' + name, str(opt)]
        if force:
            cmd[0].append('-f')
        add_opt('pus_per_node', pus_per_node)
        add_opt('work_item_size', work_item_size)
        add_opt('tasks_in_queue_per_pu', tasks_in_queue_per_pu)
        add_opt('load_workers_per_node', load_workers_per_node)
        add_opt('save_workers_per_node', save_workers_per_node)
        current_env = os.environ.copy()
        if custom_env:
            for k, v in custom_env.iteritems():
                current_env[k] = v

        start = time.time()
        p = subprocess.Popen(
            cmd[0], env=current_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        success = (rc == 0)
        if not success:
            print(so)
        return success, elapsed

    def parse_profiler_file(self, profiler_path):
        bytes_buffer = self._storage.read(profiler_path)
        offset = 0
        # Read start and end time intervals
        t, offset = read_advance('q', bytes_buffer, offset)
        start_time = t[0]
        t, offset = read_advance('q', bytes_buffer, offset)
        end_time = t[0]
        # Profilers
        profilers = defaultdict(list)
        # Load worker profilers
        t, offset = read_advance('B', bytes_buffer, offset)
        num_load_workers = t[0]
        for i in range(num_load_workers):
            prof, offset = parse_profiler_output(bytes_buffer, offset)
            profilers[prof['worker_type']].append(prof)
        # Eval worker profilers
        t, offset = read_advance('B', bytes_buffer, offset)
        num_eval_workers = t[0]
        t, offset = read_advance('B', bytes_buffer, offset)
        groups_per_chain = t[0]
        for pu in range(num_eval_workers):
            for fg in range(groups_per_chain):
                prof, offset = parse_profiler_output(bytes_buffer, offset)
                profilers[prof['worker_type']].append(prof)
        # Save worker profilers
        t, offset = read_advance('B', bytes_buffer, offset)
        num_save_workers = t[0]
        for i in range(num_save_workers):
            prof, offset = parse_profiler_output(bytes_buffer, offset)
            profilers[prof['worker_type']].append(prof)
        return (start_time, end_time), profilers

    def parse_profiler_files(self, dataset_name, job_name):
        job_path = '{}/datasets/{}/jobs/{}'.format(
            self._db_path, dataset_name, job_name)

        job = self._meta.JobDescriptor()
        job.ParseFromString(
            self._storage.read('{}/descriptor.bin'.format(job_path)))

        profilers = {}
        for n in range(job.num_nodes):
            path = '{}/profile_{}.bin'.format(job_path, n)
            time, profs = self.parse_profiler_file(path)
            profilers[n] = (time, profs)

        return profilers
