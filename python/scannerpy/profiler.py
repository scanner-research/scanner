import struct
import json
from concurrent.futures import ThreadPoolExecutor
import random
import tarfile
import os

from scannerpy.common import *
from scannerpy.protobufs import protobufs

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


class Profile:
    """
    Contains profiling information about Scanner jobs.
    """

    def __init__(self, sc, job_id, load_threads=8, subsample=None):
        self._storage = sc._storage
        job = sc._load_descriptor(protobufs.BulkJobDescriptor,
                                  'jobs/{}/descriptor.bin'.format(job_id))

        def get_prof(path, worker=True):
            file_info = self._storage.get_file_info(path)
            if file_info.file_exists:
                time, profs = self._parse_profiler_file(path, worker)
                return time, profs
            else:
                return None

        nodes = list(range(job.num_nodes))
        if subsample is not None:
            nodes = random.sample(nodes, subsample)

        with ThreadPoolExecutor(max_workers=load_threads) as executor:
            profilers = executor.map(
                get_prof, ['{}/jobs/{}/profile_{}.bin'.format(sc._db_path, job_id, n) for n in nodes])
        self._worker_profilers = {n: prof for n, prof in enumerate(profilers) if prof is not None}

        path = '{}/jobs/{}/profile_master.bin'.format(sc._db_path, job_id)
        self._master_profiler = get_prof(path, worker=False)

    def write_trace(self, path: str):
        """
        Generates a trace file in Chrome format.

        To visualize the trace, visit chrome://tracing in Google Chrome and
        click "Load" in the top left to load the trace.

        Args
        ----
        path
          Output path to write the trace.
        """

        # https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
        colors = {
            'Wait on Input Queue': 'grey',
            'Wait on Output Queue': 'grey',
        }
        traces = []
        next_tid = 0

        def make_trace_from_interval(interval, cat, proc, tid):
            trace = {
                'name': interval[0],
                'cat': cat,
                'ph': 'X',
                'ts': interval[1] / 1000,  # ns to microseconds
                'dur': (interval[2] - interval[1]) / 1000,
                'pid': proc,
                'tid': tid,
                'args': {}
            }
            if interval[0] in colors:
                trace['cname'] = colors[interval[0]]
            return trace


        if self._master_profiler is not None:
            traces.append({
                'name': 'process_name',
                'ph': 'M',
                'pid': -1,
                'args': {'name': 'Master'}
            })
            traces.append({
                'name': 'thread_name',
                'ph': 'M',
                'pid': -1,
                'tid': 0,
                'args': {'name': 'EventLoop'}
            })

            for interval in self._master_profiler[1]['intervals']:
                traces.append(make_trace_from_interval(interval, 'master', -1, 0))

        for proc, (_, worker_profiler_groups) in self._worker_profilers.items():
            traces.append({
                'name': 'process_name',
                'ph': 'M',
                'pid': proc,
                'args': {'name': 'Worker {:d}'.format(proc)}
            })
            num_load_workers = len(worker_profiler_groups['load'])
            num_eval_workers = sum([len(profs) for profs in worker_profiler_groups['eval']])
            num_stages_per_pipeline = len(worker_profiler_groups['eval'][0]) if num_eval_workers > 0 else 0
            for worker_type, profs in [('process_job',
                                        worker_profiler_groups['process_job']),
                                       ('load',
                                        worker_profiler_groups['load']),
                                       ('eval',
                                        worker_profiler_groups['eval']),
                                       ('save',
                                        worker_profiler_groups['save'])]:
                for i, prof in enumerate(profs):
                    tid = next_tid
                    next_tid += 1
                    pipeline_num = prof['worker_num']
                    tag = prof['worker_tag']

                    display_info = {
                        ('process_job', ''):  {'name': 'EventLoop',
                                               'index': lambda x: 0},
                        ('eval', 'pre'):  {'name': 'Pipeline[{:d}]:DecodeVideo',
                                           'index':
                                           lambda x: 1 + num_load_workers + x * num_stages_per_pipeline + 0},
                        ('eval', 'eval'):  {'name': 'Pipeline[{:d}]:Ops[{:d}]',
                                            'index':
                                            lambda x: 1 + num_load_workers + x * num_stages_per_pipeline + 1},
                        ('eval', 'post'):  {'name': 'Pipeline[{:d}]:EncodeVideo',
                                            'index':
                                            lambda x: 1 + num_load_workers + x * num_stages_per_pipeline + 2},
                        ('load', ''):  {'name': 'Reader[{:d}]',
                                        'index':
                                        lambda x: 1 + x},
                        ('save', ''):  {'name': 'Writer[{:d}]',
                                        'index':
                                        lambda x: 1 +  num_load_workers + num_eval_workers + x},
                    }
                    info = display_info[(worker_type, tag)]
                    name = info['name']
                    if tag == 'eval':
                        name = name.format(pipeline_num, prof['kernel_group'])
                    elif worker_type != 'process_job':
                        name = name.format(pipeline_num)
                    traces.append({
                        'name': 'thread_name',
                        'ph': 'M',
                        'pid': proc,
                        'tid': tid,
                        'args': {'name': name}
                    })
                    sort_index = info['index'](tid)
                    traces.append({
                        'name': 'thread_sort_index',
                        'ph': 'M',
                        'pid': proc,
                        'tid': tid,
                        'args': {'sort_index': sort_index}
                    })
                    for interval in prof['intervals']:
                        traces.append(make_trace_from_interval(interval, worker_type, proc, tid))


        parts = path.split('.')
        base = parts[0]
        exts = parts[1:]
        with open(base + '.trace', 'w') as f:
            f.write(json.dumps(traces))

        if exts == ['trace']:
            return path

        elif exts == ['tar', 'gz']:
            with tarfile.open(base + '.tar.gz', 'w:gz') as tar:
                tar.add(base + '.trace')
            os.remove(base + '.trace')
            return path

        else:
            raise ScannerException("Invalid trace extension '{}'. Must be .trace or .tar.gz." \
                                   .format(''.join(['.' + e for e in exts])))

    def _convert_time(self, d):
        def convert(t):
            if isinstance(t, float):
                return '{:2f}'.format(t / 1.0e9)
            return t

        return {
            k: self._convert_time(v) if isinstance(v, dict) else convert(v)
            for (k, v) in d.items()
        }

    def total_time_interval(self):
        intv, _ = list(self._worker_profilers.values())[0]
        return intv

    def statistics(self):
        totals = {}
        for (total_start,
             total_end), profiler in list(self._worker_profilers.values()):
            for kind in profiler:
                if kind not in totals:
                    totals[kind] = {}
                for thread in profiler[kind]:
                    for (key, start, end) in thread['intervals']:
                        if key not in totals[kind]:
                            totals[kind][key] = 0.0
                        totals[kind][key] += end - start
                    for (name, value) in thread['counters'].items():
                        if name not in totals[kind]:
                            totals[kind][name] = 0
                        totals[kind][name] += value

        totals['total_time'] = float(total_end - total_start)
        readable_totals = self._convert_time(totals)
        return readable_totals

    def _parse_profiler_output(self, bytes_buffer, offset):
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

        # wcrichto 12-27-18: seeing a strange issue where # of counters is corrupted in
        # output file. Putting in temporary sanity check until this is fixed.
        if num_counters > 1000000:
            num_counters = 0

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

    def _parse_profiler_file(self, profiler_path, worker=True):
        bytes_buffer = self._storage.read(profiler_path)

        offset = 0
        # Read start and end time intervals
        t, offset = read_advance('q', bytes_buffer, offset)
        start_time = t[0]
        t, offset = read_advance('q', bytes_buffer, offset)
        end_time = t[0]

        if worker:
            # Profilers
            profilers = defaultdict(list)
            # Load process_job profiler
            prof, offset = self._parse_profiler_output(bytes_buffer, offset)
            profilers[prof['worker_type']].append(prof)
            # Load worker profilers
            t, offset = read_advance('B', bytes_buffer, offset)
            num_load_workers = t[0]
            for i in range(num_load_workers):
                prof, offset = self._parse_profiler_output(bytes_buffer, offset)
                profilers[prof['worker_type']].append(prof)
            # Eval worker profilers
            t, offset = read_advance('B', bytes_buffer, offset)
            num_eval_workers = t[0]
            t, offset = read_advance('B', bytes_buffer, offset)
            groups_per_chain = t[0]
            for pu in range(num_eval_workers):
                for fg in range(groups_per_chain):
                    prof, offset = self._parse_profiler_output(
                        bytes_buffer, offset)
                    if fg > 0 and fg < groups_per_chain - 1:
                        prof['kernel_group'] = fg - 1
                    profilers[prof['worker_type']].append(prof)
            # Save worker profilers
            t, offset = read_advance('B', bytes_buffer, offset)
            num_save_workers = t[0]
            for i in range(num_save_workers):
                prof, offset = self._parse_profiler_output(bytes_buffer, offset)
                profilers[prof['worker_type']].append(prof)
            return (start_time, end_time), profilers

        else:
            return (start_time, end_time), self._parse_profiler_output(bytes_buffer, offset)[0]
