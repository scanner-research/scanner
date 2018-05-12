import struct
import json

from scannerpy.common import *


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


class Profiler:
    """
    Contains profiling information about Scanner jobs.
    """

    def __init__(self, db, job_id):
        self._storage = db._storage
        job = db._load_descriptor(db.protobufs.BulkJobDescriptor,
                                  'jobs/{}/descriptor.bin'.format(job_id))

        self._profilers = {}
        for n in range(job.num_nodes):
            path = '{}/jobs/{}/profile_{}.bin'.format(db._db_path, job_id, n)
            time, profs = self._parse_profiler_file(path)
            self._profilers[n] = (time, profs)

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
        colors = {'idle': 'grey'}
        traces = []
        next_tid = 0
        for proc, (_, worker_profiler_groups) in self._profilers.items():
            for worker_type, profs in [('load',
                                        worker_profiler_groups['load']),
                                       ('decode',
                                        worker_profiler_groups['decode']),
                                       ('eval',
                                        worker_profiler_groups['eval']),
                                       ('save',
                                        worker_profiler_groups['save'])]:
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
                            'name':
                            '{}_{:02d}_{:02d}'.format(worker_type, proc,
                                                      worker_num) +
                            ("_" + str(tag) if tag else "")
                        }
                    })
                    for interval in prof['intervals']:
                        trace = {
                            'name': interval[0],
                            'cat': worker_type,
                            'ph': 'X',
                            'ts': interval[1] / 1000,  # ns to microseconds
                            'dur': (interval[2] - interval[1]) / 1000,
                            'pid': proc,
                            'tid': tid,
                            'args': {}
                        }
                        if interval[0] in colors:
                            trace['cname'] = colors[interval[0]]
                        traces.append(trace)
        with open(path, 'w') as f:
            f.write(json.dumps(traces))

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
        intv, _ = list(self._profilers.values())[0]
        return intv

    def statistics(self):
        totals = {}
        for (total_start,
             total_end), profiler in list(self._profilers.values()):
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

    def _parse_profiler_file(self, profiler_path):
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
                profilers[prof['worker_type']].append(prof)
        # Save worker profilers
        t, offset = read_advance('B', bytes_buffer, offset)
        num_save_workers = t[0]
        for i in range(num_save_workers):
            prof, offset = self._parse_profiler_output(bytes_buffer, offset)
            profilers[prof['worker_type']].append(prof)
        return (start_time, end_time), profilers
