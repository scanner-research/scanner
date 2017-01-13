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
import tempfile
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'
import seaborn as sns
from multiprocessing import cpu_count
import toml
from PIL import Image
import numpy as np
from timeit import default_timer as now
import random

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

OPENCV_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, 'build/debug/comparison/opencv/opencv_compare')

STANDALONE_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, '../build/comparison/standalone/standalone_comparison')

PEAK_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, '../build/comparison/peak/peak_comparison')

OCV_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, '../build/comparison/ocv_decode/ocv_decode')

KERNEL_SOL_PROGRAM_PATH = os.path.join(
    SCRIPT_DIR, '../build/comparison/kernel_sol/kernel_sol')

DEVNULL = open(os.devnull, 'wb', 0)

TRACE_OUTPUT_PATH = os.path.join(SCRIPT_DIR, '{}_{}.trace')

NAIVE_COLOR = '#b0b0b0'
SCANNER_COLOR = '#F39948'
PEAK_COLOR = '#FF7169'

sns.set_style("whitegrid")

def clear_filesystem_cache():
    os.system('sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"')


def run_trial(dataset_name, pipeline_name, out_job_name, opts={}):
    print('Running trial: dataset {:s}, pipeline {:s}, '
          'out_job {:s}'.format(
              dataset_name,
              pipeline_name,
              out_job_name,
          ))

    # Clear cache
    clear_filesystem_cache()
    config_path = opts['config_path'] if 'config_path' in opts else None
    db_path = opts['db_path'] if 'db_path' in opts else None
    db = scanner.Scanner(config_path=config_path)
    if db_path is not None: db._db_path = db_path
    result, t = db.run(dataset_name, pipeline_name, out_job_name, opts)
    profiler_output = {}
    if result:
        print('Trial succeeded, took {:.3f}s'.format(t))
        profiler_output = db.parse_profiler_files(dataset_name, out_job_name)
        test_interval = profiler_output[0][0]
        t = (test_interval[1] - test_interval[0])
        t /= float(1e9)  # ns to s
    else:
        print('Trial FAILED after {:.3f}s'.format(t))
        t = -1
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


def write_trace_file(profilers, dataset, job):
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
    with open(TRACE_OUTPUT_PATH.format(dataset, job), 'w') as f:
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


def video_encoding_benchmark():
    input_video = '/bigdata/wcrichto/videos/charade_short.mkv'
    num_frames = 2878 # TODO(wcrichto): automate this
    output_video = '/tmp/test.mkv'
    video_paths = '/tmp/videos.txt'
    dataset_name = 'video_encoding'
    input_width = 1920
    input_height = 1080

    variables = {
        'scale': {
            'default': 0,
            'range': []
        },
        'crf': {
            'default': 23,
            'range': [1, 10, 20, 30, 40, 50]
        },
        'gop': {
            'default': 25,
            'range': [5, 15, 25, 35, 45]
        }
    }

    pipelines = [
        'effective_decode_rate',
        'histogram',
        'knn_patches'
    ]

    variables['scale']['default'] = '{}x{}'.format(input_width, input_height)
    for scale in [1, 2, 3, 4, 8]:
        width = input_width / scale
        height = input_height / scale
        # FFMPEG says dimensions must be multiple of 2
        variables['scale']['range'].append('{}x{}'.format(width//2 * 2,
                                                          height//2 * 2))

    command_template = """
ffmpeg -i {input} -vf scale={scale} -c:v libx264 -x264opts \
    keyint={gop}:min-keyint={gop} -crf {crf} {output}
"""

    db = scanner.Scanner()
    scanner_settings = {
        'force': True,
        'node_count': 1,
        'pus_per_node': 1,
        'work_item_size': 512
    }

    all_results = {}
    for pipeline in pipelines:
        all_results[pipeline] = {}
        for var in variables:
            all_results[pipeline][var] = {}

    for current_var in variables:
        settings = {'input': input_video, 'output': output_video}
        for var in variables:
            settings[var] = variables[var]['default']

        var_range = variables[current_var]['range']
        for val in var_range:
            settings[current_var] = val
            os.system('rm -f {}'.format(output_video))
            cmd = command_template.format(**settings)
            if os.system(cmd) != 0:
                print('Error: bad ffmpeg command')
                print(cmd)
                exit()

            result, _ = db.ingest('video', dataset_name, [output_video], {'force': True})
            if result != True:
                print('Error: failed to ingest')
                exit()

            for pipeline in pipelines:
                _, result = run_trial(dataset_name, pipeline, 'test',
                                      scanner_settings)
                stats = generate_statistics(result)
                if pipeline == 'effective_decode_rate':
                    t = stats['eval']['decode']
                elif pipeline == 'histogram':
                    t = float(stats['eval']['evaluate']) - \
                        float(stats['eval']['decode'])
                else:
                    t = float(stats['eval']['caffe:net']) + \
                        float(stats['eval']['caffe:transform_input'])

                fps = '{:.3f}'.format(num_frames / float(t))
                all_results[pipeline][current_var][val] = fps

    pprint(all_results)


def count_frames(video):
    cmd = """
    ffprobe -v error -count_frames -select_streams v:0 \
          -show_entries stream=nb_read_frames \
          -of default=nokey=1:noprint_wrappers=1 \
           {}
    """
    return int(subprocess.check_output(cmd.format(video), shell=True))


def multi_gpu_benchmark(tests, frame_counts, frame_wh):
    db_path = '/tmp/scanner_multi_gpu_db'

    db = scanner.Scanner()
    scanner_settings = {
        'db_path': db_path,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 256,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {
            'SC_JOB_NAME': 'base'
        }
    }
    dataset_name = 'multi_gpu'
    video_job = 'base'

    #num_gpus = [1]
    #num_gpus = [4]
    num_gpus = [1, 2, 4]
    #num_gpus = [2, 4]
    operations = [('histogram', 'histogram_benchmark'),
                  #('caffe', 'caffe_benchmark')]
                  ('caffe', 'caffe_benchmark'),
                  ('flow', 'flow_benchmark')]


    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op, _ in operations:
            all_results[test_name][op] = []

        #frames = count_frames(video)
        os.system('rm -rf {}'.format(db_path))
        print('Ingesting {}'.format(paths))
        # ingest data
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        if result is False:
            print('Failed to ingest')
            exit()

        scanner_settings['env']['SC_JOB_NAME'] = video_job

        for op, pipeline in operations:
            for gpus in num_gpus:
                frames = frame_counts[test_name]
                if op == 'histogram':
                    if frame_wh[test_name]['width'] == 640:
                        scanner_settings['io_item_size'] = 2048
                        scanner_settings['work_item_size'] = 1024
                    else:
                        scanner_settings['io_item_size'] = 512
                        scanner_settings['work_item_size'] = 128
                elif op == 'flow':
                    frames /= 20
                    scanner_settings['io_item_size'] = 512
                    scanner_settings['work_item_size'] = 64
                elif op == 'caffe':
                    scanner_settings['io_item_size'] = 480
                    scanner_settings['work_item_size'] = 96
                elif op == 'caffe_cpm2':
                    scanner_settings['io_item_size'] = 256
                    scanner_settings['work_item_size'] = 64

                scanner_settings['node_count'] = gpus
                print('Running {}, {} GPUS'.format(op, gpus))
                t, _ = run_trial(dataset_name, pipeline,
                                 op, scanner_settings)
                print(t, frames / float(t))
                all_results[test_name][op].append(float(frames) / float(t))

    pprint(all_results)
    return all_results


def multi_gpu_graphs(test_name, frame_counts, frame_wh, results,
                     labels_on=True):
    #matplotlib.rcParams.update({'font.size': 22})
    scale = 2.5
    w = 3.33 * scale
    h = 1.25 * scale
    fig = plt.figure(figsize=(w, h))

    if False:
        fig.suptitle(
            "Scanner Multi-GPU Scaling on {width}x{height} video".format(
                width=frame_wh[test_name]['width'],
                height=frame_wh[test_name]['height'],
            ))
    ax = fig.add_subplot(111)
    if labels_on:
        ax.set_ylabel("Speedup (over 1 GPU)")
    ax.xaxis.grid(False)

    t = test_name
    # all_results = {'mean': {'caffe': [1074.2958445786187, 2167.455212488331, 4357.563170607772],
    #                                   'flow': [95.75056483734716, 126.96566457146966, 127.75415013154019],
    #                                   'histogram': [3283.778064650782,
    #                                                                         6490.032394321538,
    #                                                                         12302.865537345728]}}
    operations = [('histogram', 'HIST'),
                  ('caffe', 'DNN'),
                  ('flow', 'FLOW')]
    num_gpus = [1, 2, 4]

    ops = [op for op, _ in operations]
    labels = [l for _, l in operations]
    x = np.arange(len(labels)) * 1.2
    ys = [[0 for _ in range(len(num_gpus))] for _ in range(len(labels))]

    for j, op in enumerate(ops):
        for i, time in enumerate(results[t][op]):
            ys[j][i] = time

    for i in range(len(num_gpus)):
        xx = x + (i*0.35)
        fps = [ys[l][i] for l, _ in enumerate(labels)]
        y = [ys[l][i] / ys[l][0] for l, _ in enumerate(labels)]
        ax.bar(xx, y, 0.3, align='center', color=SCANNER_COLOR,
                    edgecolor='none')
        for (j, xy) in enumerate(zip(xx, y)):
            if i == 2:
                xyx = xy[0]
                xyy = xy[1] + 0.1
                if labels_on:
                    ax.annotate('{:d}'.format(int(fps[j])),
                                xy=(xyx, xyy), ha='center')
            if labels_on:
                ax.annotate("{:d}".format(num_gpus[i]), xy=(xy[0], -0.30),
                            ha='center', annotation_clip=False)

    yt = [0, 1, 2, 3, 4]
    ax.set_yticks(yt)
    ax.set_yticklabels(['{:d}'.format(d) for d in yt])
    ax.set_ylim([0, 4.2])

    ax.set_xticks(x+0.3)
    ax.set_xticklabels(labels, ha='center')
    fig.tight_layout()
    #ax.xaxis.labelpad = 10
    ax.tick_params(axis='x', which='major', pad=15)
    sns.despine()

    variants = ['1 GPU', '2 GPUs', '4 GPUs']

    name = 'multigpu_' + test_name
    fig.savefig(name + '.png', dpi=600)
    fig.savefig(name + '.pdf', dpi=600, transparent=True)
    with open(name + '_results.txt', 'w') as f:
        f.write('Speedup\n')
        f.write('{:10s}'.format(''))
        for l in variants:
            f.write('{:10s} |'.format(l))
        f.write('\n')
        for i, r in enumerate(ys):
            f.write('{:10s}'.format(labels[i]))
            for n in r:
                f.write('{:10f} |'.format(n / r[0]))
            f.write('\n')

        f.write('\nFPS\n')
        f.write('{:10s}'.format(''))
        for l in variants:
            f.write('{:10s} |'.format(l))
        f.write('\n')
        for i, r in enumerate(ys):
            f.write('{:10s}'.format(labels[i]))
            for n in r:
                f.write('{:10f} |'.format(n))
            f.write('\n')
    fig.clf()


def pose_reconstruction_graphs(results):
    results = []
    plt.clf()
    plt.cla()
    plt.close()

    sns.set_style('ticks')

    scale = 4.5
    #w = 3.33 * scale
    w = 1.6261 * scale
    h = 1.246 * scale
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)

    #plt.title("Scaling comparison for {}".format(pipeline))
    #ax.set_xlabel("Sampled / Base Time")
    #ax.set_ylabel("Accuracy (%)")
    ax.grid(b=False, axis='x')

    cams_list = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120, 150, 180, 210, 240, 300, 360, 420, 480]
    times = [24.551988124847412, 25.52646803855896, 32.45369601249695, 32.8526508808136, 62.02082681655884, 63.67399311065674, 98.23960590362549, 86.65086603164673, 105.89570307731628, 139.70814990997314, 160.13016200065613, 183.25249886512756, 225.645281791687, 252.10664701461792, 307.928493976593, 403.465607881546, 501.95209407806396, 601.1797370910645]
    accuracy = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1245918367346939, 0.22112244897959168, 0.26448979591836735, 0.2866326530612246, 0.29724489795918385, 0.30755102040816334, 0.3137755102040817, 0.317857142857143, 0.3187755102040818, 0.3192857142857144], [0.20244897959183664, 0.4271428571428571, 0.5431632653061227, 0.5997959183673472, 0.6320408163265308, 0.6504081632653063, 0.6575510204081632, 0.6627551020408159, 0.6671428571428568, 0.6705102040816324], [0.32153061224489804, 0.6142857142857148, 0.7201020408163268, 0.7681632653061229, 0.799489795918367, 0.8161224489795917, 0.8325510204081632, 0.8401020408163267, 0.8480612244897958, 0.8572448979591835], [0.3009183673469387, 0.6352040816326532, 0.7828571428571425, 0.8359183673469388, 0.8756122448979592, 0.8906122448979591, 0.8973469387755104, 0.899795918367347, 0.9074489795918366, 0.9132653061224488], [0.3768367346938776, 0.7015306122448983, 0.8321428571428572, 0.8706122448979592, 0.8954081632653059, 0.907755102040816, 0.9156122448979593, 0.9173469387755102, 0.9223469387755102, 0.9273469387755107], [0.5231632653061223, 0.7931632653061221, 0.8673469387755101, 0.892448979591837, 0.9057142857142862, 0.9133673469387762, 0.9173469387755107, 0.9185714285714289, 0.918979591836735, 0.9194897959183678], [0.6140816326530617, 0.8255102040816327, 0.8821428571428571, 0.903469387755102, 0.9136734693877555, 0.9162244897959186, 0.9167346938775512, 0.917040816326531, 0.9176530612244902, 0.9177551020408167], [0.7026530612244903, 0.8578571428571428, 0.8913265306122449, 0.9046938775510205, 0.9115306122448981, 0.914693877551021, 0.9158163265306127, 0.9160204081632659, 0.9160204081632659, 0.9161224489795926], [0.7842857142857139, 0.8785714285714284, 0.9032653061224488, 0.9114285714285721, 0.9145918367346946, 0.9154081632653067, 0.9155102040816332, 0.9156122448979599, 0.9157142857142864, 0.9157142857142864], [0.8156122448979594, 0.8885714285714285, 0.9056122448979594, 0.9130612244897967, 0.9162244897959192, 0.9167346938775518, 0.9168367346938783, 0.9168367346938783, 0.9168367346938783, 0.9168367346938783], [0.8370408163265304, 0.8965306122448976, 0.9103061224489797, 0.9164285714285719, 0.919081632653062, 0.919285714285715, 0.9193877551020415, 0.9193877551020415, 0.9193877551020415, 0.9193877551020415], [0.8461224489795914, 0.9011224489795918, 0.9137755102040819, 0.9177551020408172, 0.9193877551020415, 0.919489795918368, 0.919489795918368, 0.919489795918368, 0.919489795918368, 0.919489795918368], [0.8596938775510203, 0.9010204081632657, 0.9158163265306127, 0.9187755102040822, 0.9202040816326535, 0.9202040816326535, 0.9203061224489801, 0.9204081632653066, 0.9204081632653066, 0.9204081632653066], [0.8735714285714283, 0.9056122448979592, 0.9165306122448986, 0.9198979591836739, 0.9214285714285718, 0.9215306122448983, 0.9216326530612249, 0.9216326530612249, 0.9216326530612249, 0.9216326530612249], [0.8981632653061224, 0.918979591836735, 0.9312244897959185, 0.9352040816326533, 0.9367346938775512, 0.9369387755102043, 0.9370408163265309, 0.9371428571428574, 0.9371428571428574, 0.9371428571428574], [0.936632653061225, 0.9511224489795919, 0.9562244897959188, 0.9587755102040822, 0.9602040816326536, 0.9603061224489801, 0.9603061224489801, 0.9603061224489801, 0.9603061224489801, 0.9603061224489801], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    relative_accuracy = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2200019068233354, 0.27652904040404025, 0.3046977169655742, 0.3150271335807051, 0.3237896310039169, 0.32781462585034027, 0.3314254792826222, 0.3362752525252525, 0.3407933931148216, 0.34228148835291683], [0.15539620696763548, 0.33249136775922494, 0.4272917182024324, 0.4857562358276645, 0.517957637600495, 0.5418097299525871, 0.554139713461142, 0.5633050144300144, 0.5712834982477838, 0.5785179344465059], [0.1784388785817357, 0.4091975147868003, 0.5199680200751627, 0.5769826502862219, 0.6055416309880597, 0.624973001205144, 0.6380683066933067, 0.649441253191253, 0.6611274677703248, 0.6710789448646591], [0.6832836726765299, 0.7546406569620858, 0.7854907552764695, 0.8007088268873982, 0.8108388258566832, 0.8190142793714225, 0.8235475794047221, 0.8258605719677145, 0.8296156640978067, 0.8321071864643294], [0.5453093434343433, 0.6737817747728461, 0.7152295462474031, 0.7341095491809785, 0.7436966347144918, 0.7512928777571635, 0.7602189051028337, 0.764250640232783, 0.768895679717108, 0.7761165808397947], [0.6649927314748745, 0.8000522077129217, 0.8391168295989726, 0.8529726483833625, 0.8620826415647842, 0.8666926426747859, 0.8713548594262882, 0.8746244172494173, 0.8791627598591886, 0.8824185219542361], [0.7539363790970934, 0.8752784992784993, 0.9141780560709133, 0.9307424757781901, 0.9404477942692238, 0.9461595547309838, 0.9489433621933624, 0.9502693516800662, 0.9513122294372297, 0.9523434601113174], [0.850369897959184, 0.9323852040816321, 0.9520025510204084, 0.9610331632653064, 0.9656505102040823, 0.9711352040816332, 0.9725000000000006, 0.9744260204081637, 0.9751147959183676, 0.9757780612244902], [0.921734693877551, 0.9733673469387761, 0.9877551020408175, 0.9941836734693884, 0.9958163265306126, 0.9965306122448981, 0.9971428571428572, 0.9972448979591837, 0.9973469387755103, 0.9973469387755103], [0.9433673469387758, 0.9840816326530623, 0.992755102040817, 0.9951020408163267, 0.9957142857142859, 0.9964285714285714, 0.9965306122448979, 0.9967346938775511, 0.9968367346938776, 0.9968367346938776], [0.9439795918367346, 0.9841836734693886, 0.9917346938775521, 0.9963265306122454, 0.9975510204081636, 0.9978571428571431, 0.9979591836734696, 0.9980612244897961, 0.9980612244897961, 0.9980612244897961], [0.9661224489795924, 0.9871428571428585, 0.993673469387756, 0.9969387755102045, 0.99765306122449, 0.9979591836734696, 0.9980612244897961, 0.9982653061224491, 0.9982653061224491, 0.9982653061224491], [0.9563265306122447, 0.9870408163265317, 0.9927551020408177, 0.9953061224489804, 0.9969387755102045, 0.9975510204081636, 0.99765306122449, 0.9980612244897961, 0.9980612244897961, 0.9980612244897961], [0.9606122448979595, 0.9784693877551028, 0.9814285714285722, 0.9824489795918374, 0.9834693877551026, 0.9838775510204089, 0.9841836734693884, 0.9842857142857149, 0.9842857142857149, 0.9842857142857149], [0.9529591836734701, 0.9653061224489804, 0.9738775510204094, 0.9758163265306135, 0.976122448979593, 0.9766326530612255, 0.9767346938775521, 0.9767346938775521, 0.9768367346938786, 0.9768367346938786], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    temporal_cams_list = [(30, 5), (60, 15), (90, 30), (120, 45), (150, 75), (200, 90), (240, 120)]
    temporal_times = [32.56036618550618, 43.92940409978231, 74.30371996561686, 110.34462833404541, 104.78588042259216, 119.68804597854614, 156.62090937296549]
    temporal_accuracy = [[0.045408163265306126, 0.08479591836734696, 0.12846938775510203, 0.16316326530612246, 0.18193877551020407, 0.19632653061224492, 0.2003061224489797, 0.20244897959183678, 0.20336734693877556, 0.20397959183673478], [0.20591836734693864, 0.47571428571428526, 0.6294897959183675, 0.7170408163265307, 0.772755102040816, 0.8006122448979588, 0.8128571428571424, 0.8239795918367341, 0.832857142857142, 0.8396938775510194], [0.3671428571428572, 0.6915306122448983, 0.8213265306122444, 0.8597959183673466, 0.8846938775510206, 0.89704081632653, 0.9051020408163266, 0.9068367346938776, 0.9115306122448981, 0.9165306122448981], [0.5187755102040817, 0.7906122448979588, 0.8640816326530608, 0.8889795918367348, 0.9022448979591836, 0.9093877551020411, 0.913163265306123, 0.9143877551020414, 0.9147959183673477, 0.9153061224489804], [0.658163265306123, 0.8352040816326527, 0.8817346938775512, 0.8991836734693875, 0.9087755102040821, 0.9136734693877561, 0.9151020408163272, 0.9155102040816334, 0.9156122448979599, 0.9157142857142864], [0.7041836734693883, 0.8591836734693876, 0.8921428571428571, 0.9050000000000001, 0.9119387755102042, 0.9148979591836741, 0.9158163265306127, 0.9160204081632659, 0.9160204081632659, 0.9161224489795926], [0.7842857142857139, 0.8785714285714284, 0.9033673469387754, 0.9113265306122454, 0.914489795918368, 0.9154081632653067, 0.9155102040816332, 0.9156122448979599, 0.9157142857142864, 0.9157142857142864]]

    total_cams = 480
    total_time = 4605

    x = []
    ys = [[] for _ in range(10)]
    for i in range(len(cams_list)):
        num_cams = cams_list[i]
        dnn_time = total_time * (num_cams / float(total_cams))
        time = dnn_time + times[i]
        x.append(time / (total_time + times[-1]))
        for j, acc in enumerate(accuracy[i]):
            ys[j].append(acc * 100)

    temporal_x = []
    temporal_ys = [[] for _ in range(10)]
    for i in range(len(temporal_cams_list)):
        major, minor = temporal_cams_list[i]
        num_cams = (major + minor * 14) / 15
        dnn_time = total_time * (num_cams / float(total_cams))
        time = dnn_time + times[i]
        temporal_x.append(time)
        for j, acc in enumerate(temporal_accuracy[i]):
            temporal_ys[j].append(acc * 100)

    # for y in ys:
    #     r = lambda: random.randint(0,255)
    #     c = '#%02X%02X%02X' % (r(),r(),r())
    #     ax.plot(x, y, color=c)
    ax.plot(x, ys[0], color=SCANNER_COLOR)
    ax.set_xlim([0, 1.0])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([])
    ax.set_ylim([0, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    fig.tight_layout()
    sns.despine()
    # fig.legend(['Spark', 'Scanner CPU', 'Scanner GPU'], loc='upper left')
    test = '160422_mafia2'
    fig.savefig('poseReconstruction_{}.png'.format(test), dpi=300)
    fig.savefig('poseReconstruction_{}.pdf'.format(test), dpi=300)


def run_cmd(template, settings):
    cmd = template.format(**settings)
    if os.system(cmd) != 0:
        print('Bad command: {}'.format(cmd))
        exit()


def multi_node_benchmark():
    dataset_name = 'multi_node_benchmark'
    spark_dir = '/users/wcrichto/spark-2.1.0-bin-hadoop2.7'
    videos_dir = '/users/wcrichto/videos/movies'
    videos = [
        'meanGirls.mp4',
        'anewhope.m4v',
        'brazil.mkv',
        'fightClub.mp4',
        'excalibur.mp4'
    ]

    pipelines = [
        'histogram_benchmark',
        'caffe_benchmark'
    ]

    node_counts = [1, 2]
    hosts = ','.join(['h{}.sparktest.blguest'.format(i) for i in range(node_counts[-1])])

    db = scanner.Scanner()
    scanner_settings = {
        'force': True,
        'work_item_size': 96,
        'io_item_size': 96,
        'hosts': hosts,
        'env': {
            'SC_JOB_NAME': 'base'
        }
    }

    result, _ = db.ingest(
        'video',
        dataset_name,
        ['{}/{}'.format(videos_dir, video) for video in videos],
        {'force': True})
    if result is False:
        print('Failed to ingest')
        exit()

    run_spark = '{spark_dir}/run_sparkcaffe.sh {pipeline}'
    split_video = """
ffmpeg -i {videos_dir}/{input} -vcodec copy -acodec copy -segment_time 8 \
  -f segment {videos_dir}/segments/{segment_dir}/segment%03d.mp4
"""

    total_frames = 0

    os.system('rm -f {}/segments/*'.format(videos_dir))
    for video in videos:
        segment_dir = os.path.basename(video)
        os.system('mkdir -p {}/segments/{}'.format(videos_dir, segment_dir))
        run_cmd(split_video, {
            'input': video,
            'segment_dir': segment_dir,
            'videos_dir': videos_dir
        })
        total_frames += count_frames('{}/{}'.format(videos_dir, video))

    all_results = {}
    for pipeline in pipelines:
        all_results[pipeline] = {}
        for node_count in node_counts:
            all_results[pipeline][node_count] = {}
            scanner_settings['node_count'] = node_count

            t, _ = run_trial(dataset_name, pipeline, 'test', scanner_settings)
            all_results[pipeline][node_count]['scanner'] = total_frames / t

            start = now()
            run_cmd(run_spark, {
                'spark_dir': spark_dir,
                'pipeline': pipeline
            })
            t = now() - start
            all_results[pipeline][node_count]['spark'] = total_frames / t

    pprint(all_results)

    all_results = {
        'caffe_benchmark': {1: {'scanner': 367.10523986874557,
                                    'spark': 93.16820344357546},
                                    2: {'scanner': 580.0510210459554,
                                        'spark': 92.63622769092578}},
            'histogram_benchmark': {1: {'scanner': 892.6191112787498,
                                            'spark': 208.10595095583145},
                                            2: {'scanner': 1472.5627571211041,
                                        'spark': 221.37386863726317}}}


    def bar_chart():
        plt.title("Spark vs. Scanner on 2 nodes")
        plt.xlabel("Pipeline")
        plt.ylabel("FPS")

        labels = pipelines
        x = np.arange(len(labels))
        ys = [[0, 0] for _ in range(len(labels))]

        for (i, pipeline) in enumerate(pipelines):
            values = all_results[pipeline]
            for (j, n) in enumerate(values.values()):
                ys[j][i] = n

        width = 0.3
        colors = sns.color_palette()
        for (i, y) in enumerate(ys):
            xx = x + (i*width)
            plt.bar(xx, y, width, align='center', color=colors[i])
            for (j, xy) in enumerate(zip(xx, y)):
                speedup = xy[1] / float(ys[0][j])
                plt.annotate('{} ({:.1f}x)'.format(int(xy[1]), speedup), xy=xy, ha='center')

        plt.xticks(x+width/2, labels)
        plt.legend(['Spark', 'Scanner'], loc='upper left')
        plt.tight_layout()

        plt.savefig('multinode.png', dpi=150)

    def line_chart():
        for pipeline in pipelines:
            plt.clf()
            plt.cla()
            plt.close()

            plt.title("Spark vs. Scanner")
            plt.xlabel("Number of nodes")
            plt.ylabel("FPS")
            plt.xticks(node_counts)

            x = node_counts
            values = all_results[pipeline]
            for method in ['spark', 'scanner']:
                y = [values[n][method] for n in node_counts]
                plt.plot(x, y)
                for xy in zip(x,y):
                    val = int(xy[1])
                    speedup = xy[1] / values[n]['spark']
                    if xy[0] == node_counts[0]:
                        ha = 'left'
                    elif xy[0] == node_counts[-1]:
                        ha = 'right'
                    else:
                        ha = 'center'
                    plt.annotate('{} ({:.1f}x)'.format(int(xy[1]), speedup), xy=xy, ha=ha)

            plt.tight_layout()
            plt.legend(['Spark', 'Scanner'], loc='upper left')
            plt.savefig('multinode_{}.png'.format(pipeline), dpi=150)

    line_chart()


def image_video_decode_benchmark():
    input_video = '/bigdata/wcrichto/videos/charade_short.mkv'
    num_frames = 2878 # TODO(wcrichto): automate this
    output_video = '/tmp/test.mkv'
    output_im_bmp = '/tmp/test_bmp'
    output_im_jpg = '/tmp/test_jpg'
    paths_file = '/tmp/paths.txt'
    dataset_name = 'video_encoding'
    in_job_name = scanner.Scanner.base_job_name()
    input_width = 1920
    input_height = 1080

    scales = []

    for scale in [1, 2, 3, 4, 8]:
        width = input_width / scale
        height = input_height / scale
        # FFMPEG says dimensions must be multiple of 2
        scales.append('{}x{}'.format(width//2 * 2, height//2 * 2))

    scale_template = "ffmpeg -i {input} -vf scale={scale} -c:v libx264 {output}"
    jpg_template = "ffmpeg -i {input} {output}/frame%07d.jpg"
    bmp_template = "ffmpeg -i {input} {output}/frame%07d.bmp"

    db = scanner.Scanner()
    scanner_settings = {
        'force': True,
        'node_count': 1,
        'work_item_size': 512
    }

    def run_cmd(template, settings):
        cmd = template.format(**settings)
        if os.system(cmd) != 0:
            print('Bad command: {}'.format(cmd))
            exit()

    all_results = {}
    for scale in scales:
        all_results[scale] = {}

        os.system('rm {}'.format(output_video))
        run_cmd(scale_template, {
            'input': input_video,
            'output': output_video,
            'scale': scale
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_im_bmp))
        run_cmd(bmp_template, {
            'input': output_video,
            'output': output_im_bmp
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_im_jpg))
        run_cmd(jpg_template, {
            'input': output_video,
            'output': output_im_jpg
        })

        datasets = [('video', [output_video], 'effective_decode_rate'),
                    ('image', ['{}/{}'.format(output_im_bmp, f)
                               for f in os.listdir(output_im_bmp)],
                     'image_decode_rate'),
                    ('image', ['{}/{}'.format(output_im_jpg, f)
                               for f in os.listdir(output_im_jpg)],
                     'image_decode_rate')]

        for (i, (ty, paths, pipeline)) in enumerate(datasets):
            result, _ = db.ingest(ty, dataset_name, paths, {'force': True})
            if result != True:
                print('Error: failed to ingest')
                exit()

            pus_per_node = cpu_count() if pipeline == 'image_decode_rate' else 1
            scanner_settings['pus_per_node'] = pus_per_node
            t, result = run_trial(dataset_name, in_job_name, pipeline,
                                  'test', scanner_settings)
            stats = generate_statistics(result)
            all_results[scale][i] = {
                'decode': stats['eval']['decode'],
                'io': stats['load']['io'],
                'total': t
            }

    pprint(all_results)


def disk_size(path):
    output = subprocess.check_output("du -bh {}".format(path), shell=True)
    return output.split("\t")[0]


def storage_benchmark():
    config_path = '/tmp/scanner.toml'
    output_video = '/tmp/test.mkv'
    output_video_stride = '/tmp/test_stride.mkv'
    output_images_jpg = '/tmp/test_jpg'
    output_images_bmp = '/tmp/test_bmp'
    output_images_stride = '/tmp/test_jpg_stride'
    paths_file = '/tmp/paths.txt'
    dataset_name = 'video_encoding'
    in_job_name = scanner.Scanner.base_job_name()

    video_paths = {
        'charade': '/bigdata/wcrichto/videos/charade_short.mkv',
        'meangirls': '/bigdata/wcrichto/videos/meanGirls_medium.mp4'
    }

    datasets = [(video, scale)
                for video in [('charade', 1920, 1080, 2878),
                              ('meangirls', 640, 480, 5755)]
                for scale in [1, 2, 4, 8]]

    strides = [1, 2, 4, 8]
    disks = {
        'sdd': '/data/wcrichto/db',
        'hdd': '/bigdata/wcrichto/db',
    }

    scale_template = "ffmpeg -i {input} -vf scale={scale} -c:v libx264 {output}"
    jpg_template = "ffmpeg -i {input} {output}/frame%07d.jpg"
    bmp_template = "ffmpeg -i {input} {output}/frame%07d.bmp"
    stride_template = "ffmpeg -f image2 -i {input}/frame%*.jpg {output}"

    scanner_settings = {
        'force': True,
        'node_count': 1,
        'work_item_size': 96,
        'pus_per_node': 1,
        'config_path': config_path
    }

    scanner_toml = scanner.ScannerConfig.default_config_path()
    with open(scanner_toml, 'r') as f:
        scanner_config = toml.loads(f.read())

    all_results = []
    all_sizes = []
    for ((video, width, height, num_frames), scale) in datasets:
        width /= scale
        height /= scale
        scale = '{}x{}'.format(width//2*2, height//2*2)

        os.system('rm -f {}'.format(output_video))
        run_cmd(scale_template, {
            'input': video_paths[video],
            'scale': scale,
            'output': output_video
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_images_jpg))
        run_cmd(jpg_template, {
            'input': output_video,
            'output': output_images_jpg
        })

        os.system('mkdir -p {path} && rm -f {path}/*'.format(path=output_images_bmp))
        run_cmd(bmp_template, {
            'input': output_video,
            'output': output_images_bmp
        })

        for stride in strides:
            os.system('mkdir -p {path} && rm -f {path}/*'
                      .format(path=output_images_stride))
            for frame in range(0, num_frames, stride):
                os.system('ln -s {}/frame{:07d}.jpg {}'
                          .format(output_images_jpg, frame, output_images_stride))
            os.system('rm -f {}'.format(output_video_stride))
            run_cmd(stride_template, {
                'input': output_images_stride,
                'output': output_video_stride
            })

            jobs = [
                ('orig_video', 'video', [output_video], 'effective_decode_rate', stride),
                ('strided_video', 'video', [output_video_stride], 'effective_decode_rate', 1),
                ('exploded_jpg', 'image',
                 ['{}/{}'.format(output_images_jpg, f)
                  for f in os.listdir(output_images_jpg)],
                 'image_decode_rate', stride),
                ('exploded_bmp', 'image',
                 ['{}/{}'.format(output_images_bmp, f)
                  for f in os.listdir(output_images_bmp)],
                 'image_decode_rate', stride)
            ]

            config = (video, scale, stride)
            all_sizes.append((config, {
                'orig_video': disk_size(output_video),
                'strided_video': disk_size(output_video_stride),
                'exploded_jpg': disk_size(output_images_jpg),
                'exploded_bmp': disk_size(output_images_bmp)
            }))

            for disk in disks:
                scanner_config['storage']['db_path'] = disks[disk]
                with open(config_path, 'w') as f:
                    f.write(toml.dumps(scanner_config))
                db = scanner.Scanner(config_path=config_path)
                for (job_label, ty, paths, pipeline, pipeline_stride) in jobs:
                    config = (video, scale, stride, disk, job_label)
                    print('Running test: ', config)
                    result, _ = db.ingest(ty, dataset_name, paths, {'force': True})
                    if result != True:
                        print('Error: failed to ingest')
                        exit()

                    with open('stride.txt', 'w') as f:
                        f.write(str(pipeline_stride))
                    t, result = run_trial(dataset_name, in_job_name, pipeline,
                                          'test', scanner_settings)
                    stats = generate_statistics(result)
                    all_results.append((config, {
                        'decode': stats['eval']['decode'],
                        'io': stats['load']['io'],
                        'total': t
                    }))

    print(json.dumps(all_results))
    print(json.dumps(all_sizes))


def standalone_benchmark(tests):
    output_dir = '/tmp/standalone'
    test_output_dir = '/tmp/outputs'
    paths_file = os.path.join(output_dir, 'paths.txt')

    def read_meta(path):
        files = [name for name in os.listdir(path)
                 if os.path.isfile(os.path.join(path, name))]
        filename = os.path.join(path, files[0])
        with Image.open(filename) as im:
            width, height = im.size
        return {'num_images': len(files) - 2, 'width': width, 'height': height}

    def write_meta_file(path, meta):
        with open(os.path.join(path, 'meta.txt'), 'w') as f:
            f.write(str(meta['num_images']) + '\n')
            f.write(str(meta['width']) + '\n')
            f.write(str(meta['height']))

    def write_paths(paths):
        with open(paths_file, 'w') as f:
            f.write(paths[0])
            for p in paths[1:]:
                f.write('\n' + p)

    def run_standalone_trial(input_type, paths_file, operation):
        print('Running standalone trial: {}, {}, {}'.format(
            input_type,
            paths_file,
            operation))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        p = subprocess.Popen([
            STANDALONE_PROGRAM_PATH,
            '--input_type', input_type,
            '--paths_file', paths_file,
            '--operation', operation
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    operations = ['histogram', 'flow', 'caffe']

    bmp_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.bmp"
    jpg_template = "ffmpeg -i {input} -start_number 0 {output}/frame%07d.jpg"

    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op in operations:
            all_results[test_name][op] = []

        # # bmp
        # os.system('rm -rf {}'.format(output_dir))
        # os.system('mkdir {}'.format(output_dir))
        # run_paths = []
        # for p in paths:
        #     base = os.path.basename(p)
        #     run_path = os.path.join(output_dir, base)
        #     os.system('mkdir -p {}'.format(run_path))
        #     run_cmd(bmp_template, {
        #         'input': p,
        #         'output': run_path
        #     })
        #     meta = read_meta(run_path)
        #     write_meta_file(run_path, meta)
        #     run_paths.append(run_path)
        # write_paths(run_paths)

        # for op in operations:
        #     all_results[test_name][op].append(
        #         run_standalone_trial('bmp', paths_file, op))

        # # # jpg
        # os.system('rm -rf {}'.format(output_dir))
        # os.system('mkdir {}'.format(output_dir))
        # run_paths = []
        # for p in paths:
        #     base = os.path.basename(p)
        #     run_path = os.path.join(output_dir, base)
        #     os.system('mkdir -p {}'.format(run_path))
        #     run_cmd(jpg_template, {
        #         'input': p,
        #         'output': run_path
        #     })
        #     meta = read_meta(run_path)
        #     write_meta_file(run_path, meta)
        #     run_paths.append(run_path)
        # write_paths(run_paths)

        # for op in operations:
        #     all_results[test_name][op].append(
        #         run_standalone_trial('jpg', paths_file, op))

        # video

        for op in operations:
            os.system('rm -rf {}'.format(output_dir))
            os.system('mkdir {}'.format(output_dir))

            run_paths = []
            for p in paths:
                base = os.path.basename(p)
                run_path = os.path.join(output_dir, base)
                os.system('cp {} {}'.format(p, run_path))
                run_paths.append(run_path)
            write_paths(run_paths)

            os.system('rm -rf {}'.format(test_output_dir))
            os.system('mkdir -p {}'.format(test_output_dir))
            all_results[test_name][op].append(
                run_standalone_trial('mp4', paths_file, op))

    print(all_results)
    return all_results


def scanner_benchmark(tests, wh):
    db_dir = '/tmp/scanner_db'

    db = scanner.Scanner()
    db._db_path = db_dir
    scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 256,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {}
    }
    dataset_name = 'test'
    raw_job = 'raw_job'
    jpg_job = 'jpg_job'
    video_job = 'base'

    operations = [('histogram', 'histogram_benchmark'),
                  ('flow', 'flow_benchmark'),
                  ('caffe', 'caffe_benchmark')]

    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op, _ in operations:
            all_results[test_name][op] = []

        os.system('rm -rf {}'.format(db_dir))
        # ingest data
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        assert(result)

        # pre process data into exploded raw and jpeg
        # scanner_settings['env']['SC_ENCODING'] = 'RAW'
        # result, _ = db.run(dataset_name, 'kaboom', raw_job, scanner_settings)
        # assert(result)

        # scanner_settings['env']['SC_ENCODING'] = 'JPEG'
        # result, _ = db.run(dataset_name, 'kaboom', jpg_job, scanner_settings)
        # assert(result)

        # raw
        # scanner_settings['env']['SC_JOB_NAME'] = raw_job
        # for op, pipeline in operations:
        #     total, prof = run_trial(dataset_name, pipeline, 'dummy',
        #                           scanner_settings)
        #     stats = generate_statistics(prof)
        #     all_results[test_name][op].append((total, stats))

        # # jpeg
        # scanner_settings['env']['SC_JOB_NAME'] = jpg_job
        # for op, pipeline in operations:
        #     total, prof = run_trial(dataset_name, pipeline, 'dummy',
        #                             scanner_settings)
        #     stats = generate_statistics(prof)
        #     all_results[test_name][op].append((total, stats))

        # video
        scanner_settings['env']['SC_JOB_NAME'] = video_job
        for op, pipeline in operations:
            if op == 'histogram':
                if wh[test_name]['width'] == 640:
                    scanner_settings['io_item_size'] = 2048
                    scanner_settings['work_item_size'] = 1024
                else:
                    scanner_settings['io_item_size'] = 512
                    scanner_settings['work_item_size'] = 128
            elif op == 'flow':
                scanner_settings['io_item_size'] = 128
                scanner_settings['work_item_size'] = 32
            elif op == 'caffe':
                scanner_settings['io_item_size'] = 256
                scanner_settings['work_item_size'] = 64
            elif op == 'caffe_cpm2':
                scanner_settings['io_item_size'] = 256
                scanner_settings['work_item_size'] = 64
            total, prof = run_trial(dataset_name, pipeline, op,
                                    scanner_settings)
            stats = generate_statistics(prof)
            all_results[test_name][op].append((total, stats))

    print(all_results)
    return all_results


def peak_benchmark(tests, frame_counts, wh):
    db_dir = '/tmp/scanner_db'
    input_video = '/tmp/scanner_db/datasets/test/data/0_data.bin'
    test_output_dir = '/tmp/outputs'

    db = scanner.Scanner()
    db._db_path = db_dir
    scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 256,
        'work_item_size': 64,
        'tasks_in_queue_per_pu': 3,
        'force': True,
        'env': {}
    }
    dataset_name = 'test'
    video_job = 'base'

    def run_peak_trial(frames, op, width, height):
        print('Running peak trial: {}'.format(op))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        p = subprocess.Popen([
            PEAK_PROGRAM_PATH,
            '--video_path', input_video,
            '--frames', str(frames),
            '--width', str(width),
            '--height', str(height),
            '--operation', op
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    operations = ['histogram', 'flow', 'caffe']

    all_results = {}
    for test_name, paths in tests.iteritems():
        all_results[test_name] = {}
        for op in operations:
            all_results[test_name][op] = []

        os.system('rm -rf {}'.format(db_dir))
        # ingest data
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        assert(result)

        # video
        for op in operations:
            os.system('rm -rf {}'.format(test_output_dir))
            os.system('mkdir -p {}'.format(test_output_dir))
            frames = frame_counts[test_name]
            if op == 'flow':
                frames /= 20
            all_results[test_name][op].append(
                run_peak_trial(frames, op, wh[test_name]['width'],
                               wh[test_name]['height']))

    print(all_results)
    return all_results


def decode_sol(tests, frame_count):
    db_dir = '/tmp/scanner_db'
    input_video = '/tmp/scanner_db/datasets/test/data/0_data.bin'

    db = scanner.Scanner()
    db._db_path = db_dir
    scanner_settings = {
        'db_path': db_dir,
        'node_count': 1,
        'pus_per_node': 1,
        'io_item_size': 8192,
        'work_item_size': 4096,
        'tasks_in_queue_per_pu': 4,
        'force': True,
        'env': {}
    }
    dataset_name = 'test'
    video_job = 'base'

    decode_pipeline = 'effective_decode_rate'

    def run_ocv_trial(ty, path):
        print('Running ocv trial: {}'.format(path))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        p = subprocess.Popen([
            OCV_PROGRAM_PATH,
            '--decoder', ty,
            '--path', path,
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    def run_cmd(template, settings):
        cmd = template.format(**settings)
        if os.system(cmd) != 0:
            print('Bad command: {}'.format(cmd))
            exit()

    ffmpeg_cpu_template = 'ffmpeg -vcodec h264 -i {path} -f null -'
    ffmpeg_gpu_template = 'ffmpeg -vcodec h264_cuvid -i {path} -f null -'


    all_results = {}
    for test_name, paths in tests.iteritems():
        assert(len(paths) == 1)
        path = paths[0]

        all_results[test_name] = {}

        vid_path = '/tmp/vid'

        os.system('rm -rf {}'.format(db_dir))
        os.system('cp {} {}'.format(path, vid_path))

        # ingest data
        result, _ = db.ingest('video', dataset_name, paths, scanner_settings)
        assert(result)

        if test_name == 'mean':
            scanner_settings['io_item_size'] = 8192
            scanner_settings['work_item_size'] = 2048
        if test_name == 'fight':
            scanner_settings['io_item_size'] = 2048
            scanner_settings['work_item_size'] = 512


        # Scanner decode
        total, prof = run_trial(dataset_name, decode_pipeline, 'test',
                                scanner_settings)
        all_results[test_name]['scanner'] = total

        # OCV decode
        total, _ = run_ocv_trial('cpu', vid_path)
        all_results[test_name]['opencv_cpu'] = total

        total, _ = run_ocv_trial('gpu', vid_path)
        all_results[test_name]['opencv_gpu'] = total

        # FFMPEG CPU decode
        start_time = time.time()
        run_cmd(ffmpeg_cpu_template, {'path': vid_path})
        all_results[test_name]['ffmpeg_cpu'] = time.time() - start_time

        # FFMPEG GPU decode
        start_time = time.time()
        run_cmd(ffmpeg_gpu_template, {'path': vid_path})
        all_results[test_name]['ffmpeg_gpu'] = time.time() - start_time

        print('Decode test on ', test_name)
        print("{:10s} | {:6s} | {:7s}".format('Type', 'Total', 'FPS'))
        for ty, total in all_results[test_name].iteritems():
            print("{:10s} | {:6.2f} | {:7.2f}".format(
                ty, total, frame_count[test_name] / total))

    print(all_results)
    return all_results


def kernel_sol(tests):
    def run_kernel_trial(operation, path, frames):
        print('Running kernel trial: {}'.format(path))
        clear_filesystem_cache()
        current_env = os.environ.copy()
        start = time.time()
        p = subprocess.Popen([
            KERNEL_SOL_PROGRAM_PATH,
            '--operation', operation,
            '--frames', str(frames),
            '--path', path,
        ], env=current_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        so, se = p.communicate()
        rc = p.returncode
        elapsed = time.time() - start
        timings = {}
        if rc != 0:
            print('Trial FAILED after {:.3f}s'.format(elapsed))
            print(so)
            elapsed = -1
        else:
            print('Trial succeeded, took {:.3f}s'.format(elapsed))
            for line in so.splitlines():
                if line.startswith('TIMING: '):
                    k, s, v = line[len('TIMING: '):].partition(",")
                    timings[k] = float(v)
            elapsed = timings['total']
        return elapsed, timings

    operations = ['histogram', 'flow', 'caffe']
    iters = {'histogram': 50,
             'flow': 5,
             'caffe': 30}

    all_results = {}
    for test_name, paths in tests.iteritems():
        assert(len(paths) == 1)
        path = paths[0]

        all_results[test_name] = {}

        frames = 512

        for op in operations:
            all_results[test_name][op] = run_kernel_trial(
                op, path, frames)

        print('Kernel SOL on ', test_name)
        print("{:10s} | {:6s} | {:7s}".format('Kernel', 'Total', 'FPS'))
        for ty, (total, _) in all_results[test_name].iteritems():
            tot_frames = frames * iters[ty]
            print("{:10s} | {:6.2f} | {:7.2f}".format(
                ty, total, tot_frames / (1.0 * total)))

    print(all_results)
    return all_results


def standalone_graphs(frame_counts, results):
    plt.clf()
    plt.title("Standalone perf on Charade")
    plt.ylabel("FPS")
    plt.xlabel("Pipeline")

    colors = sns.color_palette()

    x = np.arange(3)
    labels = ['caffe', 'flow', 'histogram']

    test_name = 'charade'
    tests = results[test_name]
    #for test_name, tests in results.iteritems():
    if 1:
        ys = []
        for i in range(len(tests[labels[0]])):
            y = []
            for label in labels:
                print(tests)
                frames = frame_counts[test_name]
                sec, timings = tests[label][i]
                if label == 'flow':
                    frames /= 20.0
                print(label, frames, sec, frames / sec)
                y.append(frames / sec)
            ys.append(y)

        print(ys)
        for (i, y) in enumerate(ys):
            xx = x+(i*0.3)
            plt.bar(xx, y, 0.3, align='center', color=colors[i])
            for xy in zip(xx, y):
                plt.annotate("{:.2f}".format(xy[1]), xy=xy)
                print(xy)
        plt.legend(['BMP', 'JPG', 'Video'], loc='upper left')
        plt.tight_layout()
        plt.savefig('standalone_' + test_name + '.png', dpi=150)
        plt.savefig('standalone_' + test_name + '.pdf', dpi=150)


def comparison_graphs(test_name,
                      frame_counts, wh,
                      standalone_results, scanner_results,
                      peak_results,
                      labels_on=True):
    scale = 2.5
    w = 3.33 * scale
    h = 1.25 * scale
    fig = plt.figure(figsize=(w, h))
    if False:
        fig.suptitle("Microbenchmarks on {width}x{height} video".format(
            width=wh[test_name]['width'],
            height=wh[test_name]['height']))
    ax = fig.add_subplot(111)
    if labels_on:
        plt.ylabel("Speedup (over expert)")
    ax.xaxis.grid(False)

    ops = ['histogram', 'caffe', 'flow']
    labels = ['HIST', 'DNN', 'FLOW']

    standalone_tests = standalone_results[test_name]
    scanner_tests = scanner_results[test_name]
    peak_tests = peak_results[test_name]
    #for test_name, tests in results.iteritems():
    if 1:
        ys = []

        standalone_fps = []
        scanner_fps = []
        peak_fps = []

        standalone_y = []
        scanner_y = []
        peak_y = []
        for label in ops:
            frames = frame_counts[test_name]
            if label == 'flow':
                frames /= 20.0

            peak_sec, timings = peak_tests[label][0]
            if peak_sec == -1:
                peak_fps.append(0)
            else:
                peak_fps.append(frames / peak_sec)
            peak_y.append(1.0)

            sec, timings = standalone_tests[label][0]
            if sec == -1:
                standalone_y.append(0)
                standalone_fps.append(0)
            else:
                standalone_y.append(peak_sec / sec)
                standalone_fps.append(frames / sec)

            sec, timings = scanner_tests[label][0]
            if sec == -1:
                scanner_y.append(0)
                scanner_fps.append(0)
            else:
                scanner_y.append(peak_sec / sec)
                scanner_fps.append(frames / sec)

        fps = []
        fps.append(standalone_fps)
        fps.append(scanner_fps)
        fps.append(peak_fps)

        ys.append(standalone_y)
        ys.append(scanner_y)
        ys.append(peak_y)
        print(ys)

        x = np.arange(3) * 1.2

        variants = ['Baseline', 'Scanner', 'HandOpt']

        colors = [NAIVE_COLOR, SCANNER_COLOR, PEAK_COLOR]
        for (i, y) in enumerate(ys):
            xx = x+(i*0.35)
            ax.bar(xx, y, 0.3, align='center', color=colors[i],
                   edgecolor='none')
            if i == 1:
                for k, xxx in enumerate(xx):
                    ax.annotate("{}".format(labels[k]),
                                xy=(xxx, -0.08), annotation_clip=False,
                                ha='center')
            if i == 2:
                for k, xy in enumerate(zip(xx, y)):
                    xp, yp = xy
                    yp += 0.05
                    #xp += 0.1
                    ax.annotate("{:d}".format(int(peak_fps[k])), xy=(xp, yp),
                                ha='center')
        if False:
            plt.legend(['Non-expert', 'Scanner', 'Hand-authored'],
                       loc='upper right')

        ax.set_xticks(x+0.3)
        ax.set_xticklabels(['', '', ''])
        ax.xaxis.grid(False)

        yt = [0, 0.5, 1]
        ax.set_yticks(yt)
        ax.set_yticklabels(['{:.1f}'.format(d) for d in yt])
        ax.set_ylim([0, 1.1])


        plt.tight_layout()
        sns.despine()

        name = 'comparison_' + test_name
        plt.savefig(name + '.png', dpi=150)
        plt.savefig(name + '.pdf', dpi=150)
        with open(name + '_results.txt', 'w') as f:
            f.write('Speedup\n')
            f.write('{:10s}'.format(''))
            for l in variants:
                f.write('{:10s} |'.format(l))
            f.write('\n')
            for j in range(len(ys[0])):
                f.write('{:10s}'.format(labels[j]))
                for n in ys:
                    f.write('{:10f} |'.format(n[j]))
                f.write('\n')

            f.write('\nFPS\n')
            f.write('{:10s}'.format(''))
            for l in variants:
                f.write('{:10s} |'.format(l))
            f.write('\n')
            for j in range(len(fps[0])):
                f.write('{:10s}'.format(labels[j]))
                for n in fps:
                    f.write('{:10f} |'.format(n[j]))
                f.write('\n')
        plt.clf()


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

def generate_statistics(profilers):
    totals = {}
    for _, profiler in profilers.values():
        for kind in profiler:
            if not kind in totals: totals[kind] = {}
            for thread in profiler[kind]:
                for (key, start, end) in thread['intervals']:
                    if not key in totals[kind]: totals[kind][key] = 0
                    totals[kind][key] += end-start

    readable_totals = convert_time(totals)
    return readable_totals


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


def micro_comparison_driver():
    tests = {
        'fight': ['/n/scanner/apoms/videos/fightClub_50k.mp4'],
        #'excalibur': ['/n/scanner/apoms/videos/excalibur_50k.mp4'],
        #'mean': ['/n/scanner/apoms/videos/meanGirls_50k.mp4'],
    }
    frame_counts = {'charade': 21579,
                    'fight': 50350,
                    'excalibur': 50100,
                    'mean': 50350,
    }
    frame_wh = {'charade': {'width': 1920, 'height': 1080},
                'fight': {'width': 1920, 'height': 800},
                'excalibur': {'width': 1920, 'height': 1080},
                'mean': {'width': 640, 'height': 480},
    }
    #t = 'mean'
    t = 'fight'
    if 0:
        standalone_results = standalone_benchmark(tests)
        scanner_results = scanner_benchmark(tests, frame_wh)
        peak_results = peak_benchmark(tests, frame_counts, frame_wh)
        comparison_graphs(t, frame_counts, frame_wh, standalone_results,
                          scanner_results, peak_results)
    if 1:
        #640
        t = 'mean'
        standalone_results = {'mean': {'caffe': [(128.86, {'load': 34.5, 'save': 0.19, 'transform': 56.25, 'eval': 94.17, 'net': 37.91, 'total': 128.86})], 'flow': [(42.53, {'load': 0.13, 'total': 42.53, 'setup': 0.21, 'save': 5.44, 'eval': 17.58})], 'histogram': [(13.54, {'load': 7.05, 'total': 13.54, 'setup': 0.12, 'save': 0.05, 'eval': 6.32})]}}
        scanner_results = {'mean': {'caffe': [(44.495288089, {'load': {'setup': '0.000009', 'task': '2.174472', 'idle': '173.748807', 'io': '2.138090'}, 'save': {'setup': '0.000008', 'task': '1.072224', 'idle': '117.011795', 'io': '1.065889'}, 'eval': {'task': '84.702374', 'evaluate': '83.057708', 'setup': '4.623244', 'evaluator_marshal': '1.427507', 'decode': '42.458139', 'idle': '146.444473', 'caffe:net': '34.756799', 'caffe:transform_input': '5.282478', 'memcpy': '1.353623'}})], 'flow': [(34.700563742, {'load': {'setup': '0.000010', 'task': '0.654652', 'idle': '83.952595', 'io': '0.641715'}, 'save': {'setup': '0.000008', 'task': '6.257866', 'idle': '62.448266', 'io': '6.257244'}, 'eval': {'task': '20.600713', 'evaluate': '20.410105', 'setup': '2.016671', 'evaluator_marshal': '0.094336', 'decode': '1.044027', 'idle': '105.924678', 'memcpy': '0.089637', 'flowcalc': '17.262241'}})], 'histogram': [(15.449293653, {'load': {'setup': '0.000007', 'task': '2.212311', 'idle': '83.767484', 'io': '2.192515'}, 'save': {'setup': '0.000008', 'task': '0.659185', 'idle': '59.795770', 'io': '0.658409'}, 'eval': {'task': '20.127624', 'evaluate': '19.132718', 'setup': '2.043204', 'histogram': '4.870099', 'decode': '14.261789', 'idle': '97.814596', 'evaluator_marshal': '0.845618', 'memcpy': '0.827516'}})]}}
        peak_results = {'mean': {'caffe': [(41.27, {'feed': 40.9, 'load': 0.0, 'total': 41.27, 'transform': 3.28, 'decode': 40.97, 'idle': 9.04, 'eval': 37.21, 'net': 33.92, 'save': 0.34})], 'flow': [(29.91, {'feed': 15.62, 'load': 0.0, 'total': 29.91, 'decode': 0.85, 'eval': 18.21, 'save': 7.74})], 'histogram': [(12.3, {'feed': 12.25, 'load': 0.0, 'total': 12.3, 'setup': 0.0, 'decode': 12.26, 'eval': 4.11, 'save': 0.05})]}}
        comparison_graphs(t, frame_counts, frame_wh, standalone_results,
                          scanner_results, peak_results)
    if 1:
        #1920
        t = 'fight'
        standalone_results = {'fight': {'caffe': [(252.92, {'load': 159.73, 'save': 0.19, 'transform': 55.09, 'eval': 93.0, 'net': 37.9, 'total': 252.92})], 'flow': [(57.03, {'load': 0.19, 'total': 57.03, 'setup': 0.23, 'save': 0.0, 'eval': 56.66})], 'histogram': [(52.41, {'load': 38.72, 'total': 52.41, 'setup': 0.16, 'save': 0.09, 'eval': 13.43})]}}
        scanner_results = {'fight': {'caffe': [(134.643764659, {'load': {'setup': '0.000313', 'task': '2.732178', 'idle': '483.639105', 'io': '2.694216'}, 'save': {'setup': '0.000012', 'task': '0.998610', 'idle': '327.472855', 'io': '0.992970'}, 'eval': {'task': '227.921991', 'evaluate': '193.274604', 'setup': '6.592625', 'evaluator_marshal': '34.380673', 'decode': '99.733381', 'idle': '421.211272', 'caffe:net': '34.503387', 'caffe:transform_input': '58.372106', 'memcpy': '34.279553'}})], 'flow': [(68.246963461, {'load': {'setup': '0.000659', 'task': '0.232119', 'idle': '258.826515', 'io': '0.198524'}, 'save': {'setup': '0.000007', 'task': '0.003803', 'idle': '194.370179', 'io': '0.003245'}, 'eval': {'task': '72.947458', 'evaluate': '68.481038', 'setup': '2.086686', 'evaluator_marshal': '4.323619', 'decode': '4.343175', 'idle': '300.531346', 'memcpy': '4.314552', 'flowcalc': '61.589261'}})], 'histogram': [(61.569025441, {'load': {'setup': '0.000008', 'task': '2.827806', 'idle': '266.692132', 'io': '2.804325'}, 'save': {'setup': '0.000006', 'task': '0.615977', 'idle': '182.132882', 'io': '0.613140'}, 'eval': {'task': '68.860379', 'evaluate': '67.389445', 'setup': '2.053393', 'histogram': '7.404950', 'decode': '59.979797', 'idle': '293.526000', 'evaluator_marshal': '1.278468', 'memcpy': '1.234116'}})]}}
        peak_results = {'fight': {'caffe': [(117.62, {'feed': 117.3, 'load': 0.0, 'total': 117.62, 'transform': 63.25, 'decode': 117.36, 'idle': 23.6, 'eval': 97.88, 'net': 34.63, 'save': 0.48})], 'flow': [(63.29, {'feed': 53.95, 'load': 0.0, 'total': 63.29, 'decode': 3.37, 'eval': 63.06, 'save': 2.44})], 'histogram': [(52.09, {'feed': 52.06, 'load': 0.0, 'total': 52.09, 'setup': 0.0, 'decode': 52.07, 'eval': 6.43, 'save': 0.53})]}}
        comparison_graphs(t, frame_counts, frame_wh, standalone_results,
                          scanner_results, peak_results)

    #decode_sol(tests, frame_counts)
    tests = {
        #'fight': ['/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'],
        #'excalibur': ['/n/scanner/wrichto.new/videos/movies/excalibur.mp4'],
        'mean': ['/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'],
    }
    frame_counts = {'charade': 163430,
                    'fight': 200158,
                    'excalibur': 202275,
                    'mean': 139301
    }
    if 0:
        #decode_sol(tests, frame_counts)
        kernel_sol(tests)


    tests = {
        #'fight': ['/n/scanner/wcrichto.new/videos/movies/fightClub.mp4'],
        # 'fight': [
        #     '/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'
        #     #'/n/scanner/wcrichto.new/videos/movies/private/fightClub.mp4'
        # ],
        #'excalibur': ['/n/scanner/wrichto.new/videos/movies/excalibur.mp4'],
        'mean': [
            '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4',
            '/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4'
        ],
    }
    frame_counts = {'charade': 163430,
                    'fight': 200158 * 1,
                    'excalibur': 202275,
                    'mean': 139301 * 2
    }

    t = 'mean'
    if 0:
        results = multi_gpu_benchmark(tests, frame_counts, frame_wh)
        multi_gpu_graphs(t, frame_counts, frame_wh, results)

    if 1:
        t = 'fight'
        all_results = {'fight': {'caffe': [450.6003510117739,
                                                                744.8901229071761,
                                                                1214.9085580870278],
                                            'flow': [35.26797046326607, 65.1234304140463, 111.91821397303859],
                                            'histogram': [817.7005547708027,
                                                                                   1676.5330527934939,
                                                                                   3309.0863111932586]}}
        multi_gpu_graphs(t, frame_counts, frame_wh, all_results)
    if 1:
        t = 'mean'
        results = {'mean': {'caffe': [1100.922914437792, 2188.3067699888497, 4350.245467315307],
                           'flow': [130.15578312203905, 239.4233822453851, 355.9739890240647],
                           'histogram': [3353.6737094160358,
                                                                 6694.3141921293845,
                                                                 12225.677026449643]}}
        multi_gpu_graphs(t, frame_counts, frame_wh, results)


def bench_main(args):
    out_dir = args.output_directory
    #effective_io_rate_benchmark()
    #effective_decode_rate_benchmark()
    #dnn_rate_benchmark()
    #storage_benchmark()
    micro_comparison_driver()
    # results = standalone_benchmark()
    # standalone_graphs(results)
    #multi_gpu_benchmark()


def graphs_main(args):
    graph_decode_rate_benchmark('decode_test.csv')


def trace_main(args):
    dataset = args.dataset
    job = args.job
    db = scanner.Scanner()
    db._db_path = '/tmp/scanner_multi_gpu_db'
    #db._db_path = '/tmp/scanner_db'
    profilers = db.parse_profiler_files(dataset, job)
    pprint(generate_statistics(profilers))
    write_trace_file(profilers, dataset, job)


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
