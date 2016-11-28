#!/usr/bin/env python

from __future__ import print_function
import os
import os.path
import time
import subprocess
import sys
import struct
import json
import scanner
from collections import defaultdict
from pprint import pprint
from datetime import datetime
import io
import csv
import argparse
from collections import defaultdict as dd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

LIGHTSCAN_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, 'build/debug/lightscanner')
OPENCV_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, 'build/debug/comparison/opencv/opencv_compare')

DEVNULL = open(os.devnull, 'wb', 0)

TRACE_OUTPUT_PATH = os.path.join(SCRIPT_DIR, '{}.trace')

NODES = [1]  # [1, 2, 4]
GPUS = [1, 2]  # [1, 2]  # [1, 2, 4, 8]
BATCH_SIZES = [1, 2, 4]  # [1, 2, 4, 8, 10, 12, 14, 16]  # [16, 64, 128, 256]
VIDEO_FILE = 'kcam_videos_small.txt'
BATCHES_PER_WORK_ITEM = 4
TASKS_IN_QUEUE_PER_GPU = 4
LOAD_WORKERS_PER_NODE = 2

def clear_filesystem_cache():
    os.system('sudo /sbin/sysctl vm.drop_caches=3')

def run_trial(dataset_name, in_job_name, pipeline_name, out_job_name, opts={}):
    print('Running trial: dataset {:s}, in_job {:s}, pipeline {:s}, '
          'out_job {:s}'.format(
              dataset_name,
              in_job_name,
              pipeline_name,
              out_job_name,
          ))

    # Clear cache
    clear_filesystem_cache()
    db = scanner.Scanner()
    result, t = db.run(dataset_name, in_job_name, pipeline_name, out_job_name,
                       opts)
    profiler_output = {}
    if result:
        print('Trial succeeded, took {:.3f}s'.format(t))
        profiler_output = db.parse_profiler_files(dataset_name, out_job_name)
        test_interval = profiler_output[0][0]
        t = (test_interval[1] - test_interval[0])
        t /= float(1e9)  # ns to s
    else:
        print('Trial FAILED after {:.3f}s'.format(t))
        # elapsed = -1
    return t, profiler_output


def run_opencv_trial(video_file,
                     gpus_per_node,
                     batch_size):
    print('Running opencv trial: {:d} gpus, {:d} batch size'.format(
        gpus_per_node,
        batch_size
    ))
    clear_filesystem_cache()
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


def run_caffe_trial(net,
                    net_descriptor_file,
                    device_type,
                    net_input_width,
                    net_input_height,
                    num_elements,
                    batch_size):
    print(('Running trial: {}, {}, {}, {:d}x{:d} net input, {:d} elements, '
           '{:d} batch_size').format(
               net,
               net_descriptor_file,
               device_type,
               net_input_width,
               net_input_height,
               num_elements,
               batch_size
           ))
    clear_filesystem_cache()
    current_env = os.environ.copy()
    if device_type == "CPU":
        current_env["OMP_NUM_THREADS"] = "68"
        current_env["KMP_BLOCKTIME"] = "10000000"
        current_env["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    start = time.time()
    p = subprocess.Popen([
        'build/comparison/caffe/caffe_throughput',
        '--net_descriptor_file', net_descriptor_file,
        '--device_type', device_type,
        '--net_input_width', str(net_input_width),
        '--net_input_height', str(net_input_height),
        '--num_elements', str(num_elements),
        '--batch_size', str(batch_size),
    ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''.join([line for line in p.stdout])
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    profiler_output = {}
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        print(output, file=sys.stderr)
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
    for settings, times in zip(trial_settings, trial_times):
        total_time = min([t for t in times if t != -1])
        print((' {:>8s} | {:>6s} | {:>4d}x{:<4d} | {:>5d} | {:>5d} | {:>6.3f}s '
               ' {:>8.3f}ms')
              .format(
                  settings['net'],
                  settings['device_type'],
                  settings['net_input_width'],
                  settings['net_input_height'],
                  settings['num_elements'],
                  settings['batch_size'],
                  total_time / 1000.0,
                  total_time / settings['num_elements']))


def write_trace_file(profilers, job):
    traces = []

    next_tid = 0
    for proc, (_, worker_profiler_groups) in profilers.iteritems():
        for worker_type, profs in [('load', worker_profiler_groups['load']),
                                   ('decode', worker_profiler_groups['decode']),
                                   ('eval', worker_profiler_groups['eval']),
                                   ('save', worker_profiler_groups['save'])]:
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
                        'name': '{}_{:02d}_{:02d}'.format(
                            worker_type, proc, worker_num) + (
                                "_" + str(tag) if tag else "")
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
    with open(TRACE_OUTPUT_PATH.format(job), 'w') as f:
        f.write(json.dumps(traces))


def dicts_to_csv(headers, dicts):
    output = io.BytesIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for d in dicts:
        writer.writerow(d)
    return output.getvalue()


def get_trial_total_io_read(result):
    total_time, profilers = result

    total_io = 0
    # Per node
    for node, (_, profiler) in profilers.iteritems():
        for prof in profiler['load']:
            counters = prof['counters']
            total_io += counters['io_read'] if 'io_read' in counters else 0
    return total_io

def effective_io_rate_benchmark():
    dataset_name = 'kcam_benchmark'
    in_job_name = scanner.Scanner.base_job_name()
    pipeline_name = 'effective_io_rate'
    out_job_name = 'eir_test'
    trial_settings = [{'force': True,
                       'node_count': 1,
                       'pus_per_node': 1,
                       'work_item_size': wis,
                       'load_workers_per_node': workers,
                       'save_workers_per_node': 1}
                      for wis in [64, 128, 256, 512, 1024, 2048, 4096, 8096]
                      for workers in [1, 2, 4, 8, 16]]
    results = []
    io = []
    for settings in trial_settings:
        result = run_trial(dataset_name, in_job_name, pipeline_name,
                           out_job_name, settings)
        io.append(get_trial_total_io_read(result) / (1024 * 1024)) # to mb
        results.append(result)
    rows = [{
        'work_item_size': y['work_item_size'],
        'load_workers_per_node': y['load_workers_per_node'],
        'time': x[0],
        'MB': i,
        'MB/s': i/x[0],
        'Effective MB/s': io[-1]/x[0]
    } for x, i, y in zip(results, io, trial_settings)]
    output_csv = dicts_to_csv(['work_item_size',
                               'load_workers_per_node',
                               'time',
                               'MB',
                               'MB/s',
                               'Effective MB/s'],
                              rows)
    print('Effective IO Rate Trials')
    print(output_csv)

def get_trial_total_decoded_frames(result):
    total_time, profilers = result

    total_decoded_frames = 0
    total_effective_frames = 0
    # Per node
    for node, (_, profiler) in profilers.iteritems():
        for prof in profiler['eval']:
            c = prof['counters']
            total_decoded_frames += (
                c['decoded_frames'] if 'decoded_frames' in c else 0)
            total_effective_frames += (
                c['effective_frames'] if 'effective_frames' in c else 0)
    return total_decoded_frames, total_effective_frames

def effective_decode_rate_benchmark():
    dataset_name = 'anewhope'
    in_job_name = scanner.Scanner.base_job_name()
    pipeline_name = 'effective_decode_rate'
    out_job_name = 'edr_test'
    trial_settings = [{'force': True,
                       'node_count': 1,
                       'pus_per_node': pus,
                       'work_item_size': wis,
                       'load_workers_per_node': 1,
                       'save_workers_per_node': 1}
                      for wis in [128, 256, 512, 1024, 2048, 4096]
                      for pus in [1, 2]]
    results = []
    decoded_frames = []
    for settings in trial_settings:
        result = run_trial(dataset_name, in_job_name, pipeline_name,
                           out_job_name, settings)
        decoded_frames.append(get_trial_total_decoded_frames(result))
        results.append(result)

    rows = [{
        'work_item_size': y['work_item_size'],
        'pus_per_node': y['pus_per_node'],
        'time': x[0],
        'decoded_frames': d[0],
        'effective_frames': d[1],
    } for x, d, y in zip(results, decoded_frames, trial_settings)]
    output_csv = dicts_to_csv(['work_item_size',
                               'pus_per_node',
                               'time',
                               'decoded_frames',
                               'effective_frames'],
                              rows)
    print('Effective Decode Rate Trials')
    print(output_csv)


def dnn_rate_benchmark():
    dataset_name = 'benchmark_kcam_dnn'
    in_job_name = scanner.Scanner.base_job_name()
    pipeline_name = 'dnn_rate'
    out_job_name = 'dnnr_test'

    nets = [
        ('features/squeezenet.toml', [32, 64, 128]),
        ('features/alexnet.toml', [128, 256, 512]),
        ('features/googlenet.toml', [48, 96, 192]),
        ('features/resnet.toml', [8, 16, 32]),
        ('features/fcn8s.toml', [2, 4, 6]),
    ]
    trial_settings = [{'force': True,
                       'node_count': 1,
                       'pus_per_node': 1,
                       'work_item_size': max(batch_size * 2, 128),
                       'load_workers_per_node': 4,
                       'save_workers_per_node': 4,
                       'env': {
                           'SC_NET': net,
                           'SC_BATCH_SIZE': str(batch_size),
                       }}
                      for net, batch_sizes in nets
                      for batch_size in batch_sizes]
    results = []
    decoded_frames = []
    for settings in trial_settings:
        result = run_trial(dataset_name, in_job_name, pipeline_name,
                           out_job_name, settings)
        decoded_frames.append(get_trial_total_decoded_frames(result))
        results.append(result)

    rows = [{
        'net': y['env']['SC_NET'],
        'batch_size': y['env']['SC_BATCH_SIZE'],
        'time': x[0],
        'frames': d[1],
        'ms/frame': ((x[0] * 1000) / d[1]) if d[1] != 0 else 0
    } for x, d, y in zip(results, decoded_frames, trial_settings)]
    out_csv = dicts_to_csv(['net', 'batch_size', 'time', 'frames', 'ms/frame'],
                           rows)
    print('DNN Rate Trials')
    print(out_csv)


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
    #['alex_net', 'features/alex_net.toml'],
    #['resnet', 'features/resnet.toml'],
    #['googlenet', 'features/googlenet.toml'],
    #['fcn', 'features/fcn8s.toml'],
    ['vgg', 'features/vgg.toml'],
]

def caffe_benchmark_cpu_trials():
    trial_settings = [
        {'net': nets[0][0],
         'net_descriptor_file': nets[0][1],
         'device_type': 'CPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 256,
         'batch_size': batch_size}
        for batch_size in [1, 2, 4, 8, 16, 32]
    ] + [
        {'net': net[0],
         'net_descriptor_file': net[1],
         'device_type': 'CPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 64,
         'batch_size': batch_size}
        for net in nets
        for batch_size in [1, 2, 4, 8, 16]]
    times = []
    for settings in trial_settings:
        trial_times = []
        for i in range(5):
            t = run_caffe_trial(**settings)
            trial_times.append(t)
        times.append(trial_times)

    print_caffe_trial_times('Caffe Throughput Benchmark', trial_settings, times)


def caffe_benchmark_gpu_trials():
    trial_settings = [
        {'net': nets[0][0],
         'net_descriptor_file': nets[0][1],
         'device_type': 'GPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 4096,
         'batch_size': batch_size}
        for batch_size in [1, 2, 4, 8, 16, 32]
    ] + [
        {'net': net[0],
         'net_descriptor_file': net[1],
         'device_type': 'GPU',
         'net_input_width': -1,
         'net_input_height': -1,
         'num_elements': 2048,
         'batch_size': batch_size}
        for net in nets
        for batch_size in [1, 2, 4, 8, 16, 32]]
    times = []
    for settings in trial_settings:
        trial_times = []
        for i in range(5):
            t = run_caffe_trial(**settings)
            trial_times.append(t)
        times.append(trial_times)

    print_caffe_trial_times('Caffe Throughput Benchmark', trial_settings, times)


def convert_time(d):
    def convert(t):
        return '{:2f}'.format(t / 1.0e9)
    return {k: convert_time(v) if isinstance(v, dict) else convert(v) \
            for (k, v) in d.iteritems()}


def print_statistics(profilers):
    totals = {}
    for _, profiler in profilers.values():
        for kind in profiler:
            if not kind in totals: totals[kind] = {}
            for thread in profiler[kind]:
                for (key, start, end) in thread['intervals']:
                    if not key in totals[kind]: totals[kind][key] = 0
                    totals[kind][key] += end-start

    readable_totals = convert_time(totals)
    pprint(readable_totals)


def graph_io_rate_benchmark(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    mb = 0
    wis_per_node = defaultdict(list)
    for row in rows:
        wis = int(row['work_item_size'])
        lwpn = row['load_workers_per_node']
        mbs = row['MB/s']
        embs = row['Effective MB/s']
        mb = row['MB']
        wis_per_node[wis].append([lwpn, mbs, embs])

    wis = [64, 128, 256, 512, 1024, 2048, 4096, 8096]
    colors = ['g', 'b', 'k', 'w', 'm', 'c', 'r', 'y']
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    for w, c in zip(wis, colors):
        d = wis_per_node[w]
        print(d)
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[1], d),
                color=c,
                linestyle='--')
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[2], d),
                color=c,
                linestyle='-',
                label=str(w) + ' wis')

    ax.set_xlabel('Load threads')
    ax.set_ylabel('MB/s')
    ax.legend()

    #ax.set_title('Loading ' + mb + ' MB on bodega SSD')
    #plt.savefig('io_rate_bodega.png', dpi=150)
    ax.set_title('Loading ' + mb + ' MB on GCS')
    plt.savefig('io_rate_gcs.png', dpi=150)


def graph_decode_rate_benchmark(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    wis_per_node = defaultdict(list)
    for row in rows:
        print(row)
        wis = int(row['work_item_size'])
        pus = int(row['pus_per_node'])
        t = float(row['time'])
        df = int(row['decoded_frames'])
        ef = int(row['effective_frames'])
        wis_per_node[wis].append([pus, t, df, ef])

    #wis = [64, 128, 256, 512, 1024, 2048]
    wis = [128, 256, 512, 1024, 2048, 4096]
    colors = ['g', 'b', 'k', 'y', 'm', 'c', 'r', 'w']
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    for w, c in zip(wis, colors):
        d = wis_per_node[w]
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[2]/x[1], d),
                color=c,
                linestyle='--')
        ax.plot(map(lambda x: x[0], d),
                map(lambda x: x[3]/x[1], d),
                color=c,
                linestyle='-',
                label=str(w) + ' wis')

    ax.set_xlabel('PUs')
    ax.set_ylabel('Decode FPS')
    ax.legend()

    ax.set_title('Decoding frames on Intel')
    plt.savefig('decode_rate_intel.png', dpi=150)


def bench_main(args):
    out_dir = args.output_directory
    #effective_io_rate_benchmark()
    effective_decode_rate_benchmark()
    #dnn_rate_benchmark()


def graphs_main(args):
    graph_decode_rate_benchmark('decode_test.csv')


def trace_main(args):
    dataset = args.dataset
    job = args.job
    db = scanner.Scanner()
    profilers = db.parse_profiler_files(dataset, job)
    print_statistics(profilers)
    write_trace_file(profilers, job)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Perform profiling tasks')
    subp = p.add_subparsers(help='sub-command help')
    # Bench
    bench_p = subp.add_parser('bench', help='Run benchmarks')
    bench_p.add_argument('output_directory', type=str,
                         help='Where to output results')
    bench_p.set_defaults(func=bench_main)
    # Graphs
    graphs_p = subp.add_parser('graphs', help='Generate graphs from bench')
    graphs_p.set_defaults(func=graphs_main)
    # Trace
    trace_p = subp.add_parser('trace', help='Generate trace files')
    trace_p.add_argument('dataset', type=str, help='Dataset to generate trace for')
    trace_p.add_argument('job', type=str, help='Job to generate trace for')
    trace_p.set_defaults(func=trace_main)

    args = p.parse_args()
    args.func(args)
