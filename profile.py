#!/usr/bin/env python

from __future__ import print_function
import os.path
import time
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PROGRAM_PATH = os.path.join(SCRIPT_DIR, 'build/debug/lightscanner')

DEVNULL = open(os.devnull, 'wb', 0)


NODES = [1, 2, 4]
GPUS = [1, 2, 4, 8]
BATCH_SIZES = [16, 64, 128, 256]


def run_trial(node_count, gpus_per_node, batch_size):
    current_env = os.environ.copy()
    start = time.time()
    p = subprocess.Popen([PROGRAM_PATH, str(gpus_per_node), str(batch_size)],
                         env=current_env,
                         stdout=DEVNULL,
                         stderr=subprocess.STDOUT)
    pid, rc, ru = os.wait4(p.pid, 0)
    elapsed = time.time() - start
    if rc != 0:
        elapsed = -1
    return elapsed


def main(args):
    batch_size_trial_settings = [[nodes, gpus, batch_size]
                                 for nodes in [1]
                                 for gpus in GPUS
                                 for batch_size in BATCH_SIZES]
    batch_size_times = []
    for settings in batch_size_trial_settings:
        t = run_trial(settings[0], settings[1], settings[2])
        batch_size_times.append(t)

    fastest_batch_size = 1
    fastest_batch_size_time = float('Inf')
    print('         Batch Size trials           ')
    print(' =================================== ')
    print(' Nodes | GPUs/n | Batch |       Time ')
    for settings, t in zip(batch_size_trial_settings, batch_size_times):
        if (t < fastest_batch_size_time):
            fastest_batch_size = settings[2]
            fastest_batch_size_time = t
        print(' {:>5d} | {:>6d} | {:>5d} | {:>9.3f}s '.format(
            settings[0],
            settings[1],
            settings[2],
            t))
    print('Fastest batch size: {:>4d}, {:>4.3f}s'.format(
        fastest_batch_size,
        fastest_batch_size_time))

    trial_settings = [[nodes, gpus, batch_size]
                      for nodes in NODES
                      for gpus in GPUS
                      for batch_size in [fastest_batch_size]]
    times = []
    for settings in trial_settings:
        t = run_trial(settings[0], settings[1], settings[2])
        times.append(t)

    print('         Node count trials           ')
    print(' =================================== ')
    print(' Nodes | GPUs/N | Batch |       Time ')
    for settings, t in zip(trial_settings, times):
        print(' {:>5d} | {:>6d} | {:>5d} | {:>9.3f}s '.format(
            settings[0],
            settings[1],
            settings[2],
            t))


if __name__ == '__main__':
    main({})
