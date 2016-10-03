#!/usr/bin/env python

from __future__ import print_function
import os
import os.path
import time
import subprocess
import sys
import struct
import json
import re
from collections import defaultdict
from pprint import pprint

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

LIGHTSCAN_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, 'build/debug/lightscanner')
OPENCV_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, 'build/debug/comparison/opencv/opencv_compare')

DEVNULL = open(os.devnull, 'wb', 0)

TRACE_OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'profile.trace')

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


def parse_profiler_files(job_name):
    r = re.compile('^{}_job_profiler_(\d+).bin$'.format(job_name))
    files = []
    for f in os.listdir('.'):
        matches = r.match(f)
        if matches is not None:
            files.append(int(matches.group(1)))

    files.sort()
    profilers = {}
    for n in files:
        path = '{}_job_profiler_{}.bin'.format(job_name, n)
        _, profs = parse_profiler_file(path)
        profilers[n] = profs

    return profilers


def parse_profiler_file(profiler_path):
    with open(profiler_path, 'rb') as f:
        bytes_buffer = f.read()
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
    # Decode worker profilers
    t, offset = read_advance('B', bytes_buffer, offset)
    num_decode_workers = t[0]
    for i in range(num_decode_workers):
        prof, offset = parse_profiler_output(bytes_buffer, offset)
        profilers[prof['worker_type']].append(prof)
    # Eval worker profilers
    t, offset = read_advance('B', bytes_buffer, offset)
    num_eval_workers = t[0]
    for i in range(num_eval_workers):
        prof, offset = parse_profiler_output(bytes_buffer, offset)
        profilers[prof['worker_type']].append(prof)
    # Save worker profilers
    t, offset = read_advance('B', bytes_buffer, offset)
    num_save_workers = t[0]
    for i in range(num_save_workers):
        prof, offset = parse_profiler_output(bytes_buffer, offset)
        profilers[prof['worker_type']].append(prof)
    return (start_time, end_time), profilers


def run_trial(job_name,
              dataset_name,
              net_descriptor_file,
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
        LIGHTSCAN_PROGRAM_PATH,
        '--gpus_per_node', str(gpus_per_node),
        '--batch_size', str(batch_size),
        '--batches_per_work_item', str(batches_per_work_item),
        '--tasks_in_queue_per_gpu', str(tasks_in_queue_per_gpu),
        '--load_workers_per_node', str(load_workers_per_node),
        'run', job_name, dataset_name, net_descriptor_file,
    ], env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    profiler_output = {}
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        # elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
        test_interval, profiler_output = parse_profiler_file(job_name)
        elapsed = (test_interval[1] - test_interval[0])
        elapsed /= float(1000000000)  # ns to s
    return elapsed, profiler_output


def run_opencv_trial(video_file,
                     gpus_per_node,
                     batch_size):
    print('Running opencv trial: {:d} gpus, {:d} batch size'.format(
        gpus_per_node,
        batch_size
    ))
    current_env = os.environ.copy()
    start = time.time()
    p = subprocess.Popen([
        OPENCV_PROGRAM_PATH,
        '--video_paths_file', video_file,
        '--gpus_per_node', str(gpus_per_node),
        '--batch_size', str(batch_size)
    ], env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
    return elapsed


def run_caffe_trial(net_descriptor_file,
                    device_type,
                    net_input_width,
                    net_input_height,
                    num_elements,
                    batch_size):
    print(('Running trial: {}, {}, {:d}x{:d} net input, {:d} elements, '
           '{:d} batch_size').format(
               net_descriptor_file,
               device_type,
               net_input_width,
               net_input_height
               num_elements,
               batch_size
           ))
    current_env = os.environ.copy()
    start = time.time()
    p = subprocess.Popen([
        'build/comparison/caffe/caffe_throughput'
        '--net_descriptor_file', net_descriptor_file,
        '--device_type', device_type,
        '--net_input_width', str(net_input_width),
        '--net_input_height', str(net_input_height),
        '--num_elements', str(num_elements),
        '--batch_size', str(batch_size),
    ], env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    profiler_output = {}
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        # elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
        elapsed *= float(1000)  # s to ms
    return elapsed


def print_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' =========================================================== ')
    print(' Nodes | GPUs/n | Batch | Loaders | Total Time | Eval Time ')
    for settings, t in zip(trial_settings, trial_times):
        total_time = t[0]
        eval_time = 0
        for prof in t[1]['eval']:
            for interval in prof['intervals']:
                if interval[0] == 'task':
                    eval_time += interval[2] - interval[1]
        eval_time /= float(len(t[1]['eval']))
        eval_time /= float(1000000000)  # ns to s
        print(' {:>5d} | {:>6d} | {:>5d} | {:>5d} | {:>9.3f}s | {:>9.3f}s '
              .format(
                  settings['node_count'],
                  settings['gpus_per_node'],
                  settings['batch_size'],
                  settings['load_workers_per_node'],
                  total_time,
                  eval_time))


def print_opencv_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' =========================================================== ')
    print(' Nodes | GPUs/n | Batch | Loaders | Total Time ')
    for settings, t in zip(trial_settings, trial_times):
        total_time = t[0]
        print(' {:>5d} | {:>6d} | {:>5d} | {:>5d} | {:>9.3f}s '
              .format(
                  1,
                  settings['gpus_per_node'],
                  settings['batch_size'],
                  1,
                  total_time))


def print_caffe_trial_times(title, trial_settings, trial_times):
    print(' {:^58s} '.format(title))
    print(' ================================================================= ')
    print(' Net      | Device |    WxH    | Elems | Batch | Time   | ms/frame ')
    for settings, t in zip(trial_settings, trial_times):
        total_time = t[0]
        print((' {:>8s} | {:>6s} | {:>4d}x{:<4d} | {:>5d} | {:>5d} | {:>6.3f}s '
               ' {:>8.3fs}ms')
              .format(
                  settings['net'],
                  settings['device_type'],
                  settings['net_input_width'],
                  settings['net_input_height'],
                  settings['num_elements'],
                  settings['batch_size'],
                  total_time / 1000.0,
                  total_time / settings['num_elements']))


def write_trace_file(profilers):
    traces = []

    next_tid = 0
    for proc, worker_profiler_groups in profilers.iteritems():
        for worker_type, profs in [('load', worker_profiler_groups['load']),
                                   ('decode', worker_profiler_groups['decode']),
                                   ('eval', worker_profiler_groups['eval']),
                                   ('save', worker_profiler_groups['save'])]:
            for i, prof in enumerate(profs):
                tid = next_tid
                next_tid += 1
                traces.append({
                    'name': 'thread_name',
                    'ph': 'M',
                    'pid': proc,
                    'tid': tid,
                    'args': {
                        'name': '{}{:02d}_{:02d}'.format(worker_type, proc, i)
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
    with open(TRACE_OUTPUT_PATH, 'w') as f:
        f.write(json.dumps(traces))


def load_workers_trials():
    trial_settings = [{'video_file': 'kcam_videos_small.txt',
                       'node_count': 1,
                       'gpus_per_node': gpus,
                       'batch_size': 64,
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


def opencv_reference_trials():
    trial_settings = [{'video_file': 'kcam_videos_small.txt',
                       'gpus_per_node': gpus,
                       'batch_size': 64}
                      for gpus in [1, 2, 4, 8]]
    times = []
    for settings in trial_settings:
        t = run_opencv_trial(**settings)
        times.append(t)

    print_opencv_trial_times(
        'OpenCV reference trials',
        trial_settings,
        times)


def single_node_scaling_trials():
    trial_settings = [{'job_name': 'single_node_scaling_trial',
                       'dataset_name': 'kcam_30',
                       'net_descriptor_file': 'features/alex_net.toml',
                       'node_count': 1,
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
        'Single-node scaling trials',
        trial_settings,
        times)


def multi_node_scaling_trials():
    trial_settings = [{'job_name': 'multi_node_scaling_trial',
                       'dataset_name': 'kcam_all',
                       'net_descriptor_file': 'features/alex_net.toml',
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
        'Multi-node scaling trials',
        trial_settings,
        times)


nets = [
    {'alex_net': 'features/alex_net.toml'},
    {'resnet': 'features/resnet.toml'},
    {'googlenet': 'features/googlenet.toml'},
    {'fcn': 'features/fcn8s.toml'},
    {'vgg': 'features/vgg.toml'},
]

def caffe_benchmark_cpu_trials():
    trial_settings = [{'net': net,
                       'net_descriptor_file': net_descriptor_file,
                       'device_type': 'CPU',
                       'net_input_width': width,
                       'net_input_height': height,
                       'num_elements': 48,
                       'batch_size': batch_size}
                      for net, net_descriptor_file in nets.iteritems(),
                      for batch_size in [1, 2, 4, 8, 16]]
    times = []
    for settings in trial_settings:
        t = run_trial(**setings)
        times.append(t)

    print_caffe_trial_times('Caffe Throughput Benchmark', trial_settings, times)


def caffe_benchmark_gpu_trials():
    pass

def main(args):
    profilers = parse_profiler_files('big')
    write_trace_file(profilers)
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(profilers['load'])


if __name__ == '__main__':
    main({})
