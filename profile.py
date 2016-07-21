#!/usr/bin/env python

from __future__ import print_function
import os.path
import time
import subprocess
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PROGRAM_PATH = os.path.join(SCRIPT_DIR, 'build/debug/lightscanner')

DEVNULL = open(os.devnull, 'wb', 0)


NODES = [1, 2, 4]
GPUS = [1, 2, 4, 8]
BATCH_SIZES = [16, 64, 128, 256]

VIDEO_FILE = 'kcam_videos_small.txt'
BATCHES_PER_WORK_ITEM = 4
TASKS_IN_QUEUE_PER_GPU = 4
LOAD_WORKERS_PER_NODE = 2


def run_trial(node_count,
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
        '--video_paths_file', VIDEO_FILE,
        '--gpus_per_node', str(gpus_per_node),
        '--batch_size', str(batch_size),
        '--batches_per_work_item', str(BATCHES_PER_WORK_ITEM),
        '--tasks_in_queue_per_gpu', str(TASKS_IN_QUEUE_PER_GPU),
        '--load_workers_per_node', str(LOAD_WORKERS_PER_NODE)
    ], env=current_env, stdout=DEVNULL, stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    if rc != 0:
        print('Trial FAILED after {:.3f}s'.format(elapsed))
        elapsed = -1
    else:
        print('Trial succeeded, took {:.3f}s'.format(elapsed))
    return elapsed


def print_trial_times(title, trial_settings, trial_times):
    print(' {:^53s} '.format(title))
    print(' ===================================================== ')
    print(' Nodes | GPUs/n | Batch | Loaders | Total Time ')
    for settings, t in zip(trial_settings, trial_times):
        print(' {:>5d} | {:>6d} | {:>5d} | {:>5d} | {:>9.3f}s'
              .format(
                  settings['node_count'],
                  settings['gpus_per_node'],
                  settings['batch_size'],
                  settings['load_workers_per_node'],
                  t))


def load_workers_trials():
    trial_settings = [{'node_count': 1,
                       'gpus_per_node': gpus,
                       'batch_size': 256,
                       'batches_per_work_item': 4,
                       'tasks_in_queue_per_gpu': 3,
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
                       'tasks_in_queue_per_gpu': 3,
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
