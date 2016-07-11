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


def run_trial(node_count, gpus_per_node, batch_size):
    print('Running trial: {:d} nodes, {:d} gpus, {:d} batch size'.format(
        node_count,
        gpus_per_node,
        batch_size
    ))
    current_env = os.environ.copy()
    start = time.time()
    p = subprocess.Popen(['mpirun', '-n', str(node_count),
                          PROGRAM_PATH, str(gpus_per_node), str(batch_size)],
                         env=current_env,
                         stdout=DEVNULL,
                         stderr=subprocess.STDOUT)
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
    print(' Nodes | GPUs/n | Batch | Normalized Time | Total Time ')
    for settings, t in zip(trial_settings, trial_times):
        num_nodes = settings[0]
        num_gpus = settings[1]
        batch_size = settings[2]
        normalized_t = t / (num_nodes * num_gpus)
        print(' {:>5d} | {:>6d} | {:>5d} | {:>14.3f}s | {:>9.3f}s'.format(
            num_nodes,
            num_gpus,
            batch_size,
            normalized_t,
            t))


def main(args):
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

    fastest_batch_sizes = defaultdict(lambda: 1)
    fastest_batch_size_times = defaultdict(lambda: float('Inf'))
    for settings, t in zip(batch_size_trial_settings, batch_size_times):
        num_nodes = settings[0]
        num_gpus = settings[1]
        batch_size = settings[2]
        normalized_t = t / (num_nodes * num_gpus)
        if (t != -1 and normalized_t < fastest_batch_size_times[num_gpus]):
            fastest_batch_sizes[num_gpus] = batch_size
            fastest_batch_size_times[num_gpus] = normalized_t

    for gpu in GPUS:
        print('Fastest batch size for {:d} GPUs: {:>4d}, {:>4.3f}s'.format(
            gpu,
            fastest_batch_sizes[gpu],
            fastest_batch_size_times[gpu]))

    trial_settings = [[nodes, gpus, batch]
                      for nodes in NODES
                      for gpus in GPUS
                      for batch in [fastest_batch_sizes[gpus]]]
    times = []
    for settings in trial_settings:
        t = run_trial(settings[0], settings[1], settings[2])
        times.append(t)

    print_trial_times('Node count trials', trial_settings, times)

if __name__ == '__main__':
    main({})
