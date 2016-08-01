#!/usr/bin/env python

from __future__ import print_function
import os.path
import time
import subprocess
import sys
import struct
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PROGRAM_PATH = os.path.join(SCRIPT_DIR, 'build/debug/lightscanner')

DEVNULL = open(os.devnull, 'wb', 0)


NODES = [1]  # [1, 2, 4]
GPUS = [1, 2]  # [1, 2]  # [1, 2, 4, 8]
BATCH_SIZES = [1, 2, 4]  # [1, 2, 4, 8, 10, 12, 14, 16]  # [16, 64, 128, 256]
VIDEO_FILE = 'kcam_videos_small.txt'
BATCHES_PER_WORK_ITEM = 4
TASKS_IN_QUEUE_PER_GPU = 4
LOAD_WORKERS_PER_NODE = 2


def read_advance(fmt, buf, offset):
    new_offset = offset + struct.calcsize(fmt)
    return struct.unpack_from(fmt, buf, offset), new_offset


def unpack_string(buf, offset):
    s = ''
    while True:
        t, offset = read_advance('B', buf, offset)
        if t == 0:
            break
        s += t
    return s, offset


def parse_profiler_output(byte_buffer, offset):
    # Node
    t, offset = read_advance('q', bytes_buffer, offset)
    node = t[0]
    # Worker type name
    worker_type, offset = unpack_string(bytes_buffer, offset)
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

    return {
        'node': node,
        'worker_type': worker_type,
        'worker_num': worker_num,
        'intervals': intervals
    }, offset


def parse_profiler_file():
    with open('profiler_0.bin', 'rb') as f:
        bytes_buffer = f.read()
    offset = 0
    profilers = defaultdict(list)
    # Load worker profilers
    t, offset = read_advance('B', bytes_buffer, offset)
    num_load_workers = t[0]
    for i in range(num_load_workers):
        prof, offset = parser_profiler_output(bytes_buffer, offset)
        profilers[prof['worker_type']].append(prof)
    # Decode worker profilers
    t, offset = read_advance('B', bytes_buffer, offset)
    num_decode_workers = t[0]
    for i in range(num_decode_workers):
        prof, offset = parser_profiler_output(bytes_buffer, offset)
        profilers[prof['worker_type']].append(prof)
    # Eval worker profilers
    t, offset = read_advance('B', bytes_buffer, offset)
    num_eval_workers = t[0]
    for i in range(num_eval_workers):
        prof, offset = parser_profiler_output(bytes_buffer, offset)
        profilers[prof['worker_type']].append(prof)
    return profilers


def run_trial(video_file,
              node_count,
              gpus_per_node,
              batch_size,
              batches_per_work_item,
              tasks_in_queue_per_gpu,
              load_workers_per_node):
    print('Running trial: {:d} nodes, {:d} gpus, {:d} batch size'.format(
        node_count,
        gpus_per_node,
        batch_size
    ))
    current_env = os.environ.copy()
    start = time.time()
    p = subprocess.Popen([
        'mpirun',
        '-n', str(node_count),
        '--bind-to', 'none',
        PROGRAM_PATH,
        '--video_paths_file', video_file,
        '--gpus_per_node', str(gpus_per_node),
        '--batch_size', str(batch_size),
        '--batches_per_work_item', str(batches_per_work_item),
        '--tasks_in_queue_per_gpu', str(tasks_in_queue_per_gpu),
        '--load_workers_per_node', str(load_workers_per_node)
    ], env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    profiler_output = {}
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        # elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
        profiler_output = parse_profiler_output()
    return elapsed, profiler_output


def print_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' =========================================================== ')
    print(' Nodes | GPUs/n | Batch | Loaders | Total Time | Decode Time ')
    for settings, t in zip(trial_settings, trial_times):
        total_time = t[0]
        decode_time = 0
        for interval in t[1]['decode'][0]['intervals']:
            if interval[0] == 'task':
                decode_time += interval[2] - interval[1]
        decode_time /= 1000000000  # ns to s
        print(' {:>5d} | {:>6d} | {:>5d} | {:>5d} | {:>9.3f}s | {:>9.3f}s '
              .format(
                  settings['node_count'],
                  settings['gpus_per_node'],
                  settings['batch_size'],
                  settings['load_workers_per_node'],
                  total_time,
                  decode_time))


def load_workers_trials():
    trial_settings = [{'video_file': 'kcam_videos_small.txt',
                       'node_count': 1,
                       'gpus_per_node': gpus,
                       'batch_size': 256,
                       'batches_per_work_item': 4,
                       'tasks_in_queue_per_gpu': 4,
                       'load_workers_per_node': workers}
                      for gpus in [1, 2, 4, 8]
                      for workers in [1, 2, 4, 8, 16]]
    times = []
    for settings in trial_settings:
        t = run_trial(**settings)
        times.append(t)

    print_trial_times(
        'Load workers trials',
        trial_settings,
        times)


def multi_node_trials():
    trial_settings = [{'video_file': 'kcam_videos.txt',
                       'node_count': nodes,
                       'gpus_per_node': gpus,
                       'batch_size': 256,
                       'batches_per_work_item': 4,
                       'tasks_in_queue_per_gpu': 4,
                       'load_workers_per_node': workers}
                      for nodes in [1, 2, 4]
                      for gpus, workers in zip([4, 8], [8, 16])]
    times = []
    for settings in trial_settings:
        t = run_trial(**settings)
        times.append(t)

    print_trial_times(
        'Multi node trials',
        trial_settings,
        times)


def work_item_size_trials():
    trial_settings = [{'node_count': 1,
                       'gpus_per_node': gpus,
                       'batch_size': 256,
                       'batches_per_work_item': work_item_size,
                       'tasks_in_queue_per_gpu': 4,
                       'load_workers_per_node': workers}
                      for gpus, workers in zip([1, 2, 4, 8], [2, 4, 8, 16])
                      for work_item_size in [4, 8, 16, 32, 64, 128, 256]]
    times = []
    for settings in trial_settings:
        t = run_trial(**settings)
        times.append(t)

    print_trial_times(
        'Work item trials',
        trial_settings,
        times)


def batch_size_trials():
    batch_size_trial_settings = [[nodes, gpus, batch]
                                 for nodes in [1]
                                 for gpus in GPUS
                                 for batch in BATCH_SIZES]
    batch_size_times = []
    for settings in batch_size_trial_settings:
        t = run_trial(settings[0], settings[1], settings[2])
        batch_size_times.append(t)

    print_trial_times(
        'Batch size trials',
        batch_size_trial_settings,
        batch_size_times)


def scaling_trials():
    trial_settings = [{'node_count': 1,
                       'gpus_per_node': gpus,
                       'batch_size': 256,
                       'batches_per_work_item': 4,
                       'tasks_in_queue_per_gpu': 4,
                       'load_worker_per_node': workers}
                      for gpus, workers in zip([1, 2, 4, 8], [1, 2, 4, 8])]
    times = []
    for settings in trial_settings:
        t = run_trial(**settings)
        times.append(t)

    print_trial_times(
        'Scaling trials',
        trial_settings,
        times)


def main(args):
    load_workers_trials()


if __name__ == '__main__':
    main({})
