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


def clear_filesystem_cache():
    os.system('sudo /sbin/sysctl vm.drop_caches=3')


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


def multi_gpu_benchmark():
    dataset_name = 'multi_gpu'

    videos = [
        '/n/scanner/wcrichto.new/videos/meanGirls.mp4'
    ]

    pipelines = [
        'effective_decode_rate',
        'knn',
        'histogram',
        'opticalflow'
    ]

    num_gpus = [1, 2, 4]

    db = scanner.Scanner()
    scanner_settings = {
        'force': True,
        'work_item_size': 24,
        'io_item_size': 96
    }

    all_results = {}
    for video in videos:
        print('Counting {}'.format(video))
        frames = count_frames(video)
        all_results[video] = {}

        print('Ingesting {}'.format(video))
        result, _ = db.ingest('video', dataset_name, [video], {'force': True})
        if result is False:
            print('Failed to ingest')
            exit()

        for pipeline in pipelines:
            all_results[video][pipeline] = {}

            for gpus in num_gpus:
                scanner_settings['node_count'] = gpus
                print('Running {}, {} GPUS'.format(pipeline, gpus))
                t, result = run_trial(dataset_name, pipeline,
                                      'test', scanner_settings)
                if result is False:
                    print('Trial failed')
                    exit()

                print(t, frames/float(t))

                all_results[video][pipeline][gpus] = frames / float(t)

    plt.title("Multi-GPU scaling in Scanner")
    plt.xlabel("Pipeline")
    plt.ylabel("FPS")

    labels = pipelines
    x = np.arange(len(labels))
    ys = [[0 for _ in range(len(num_gpus))] for _ in range(len(labels))]

    for (i, values) in enumerate(all_results[videos[0]].values()):
        for (j, n) in enumerate(values.values()):
            ys[j][i] = n

    colors = sns.color_palette()
    for (i, y) in enumerate(ys):
        xx = x + (i*0.3)
        plt.bar(xx, y, 0.3, align='center', color=colors[i])
        for (j, xy) in enumerate(zip(xx, y)):
            speedup = xy[1] / float(ys[0][j])
            plt.annotate('{} ({:.2f}x)'.format(int(xy[1]), speedup), xy=xy, ha='center')

    plt.xticks(x+0.3, labels)
    plt.legend(['{} GPUs'.format(n) for n in num_gpus], loc='upper left')

    plt.savefig('multigpu.png', dpi=150)


def run_cmd(template, settings):
    cmd = template.format(**settings)
    if os.system(cmd) != 0:
        print('Bad command: {}'.format(cmd))
        exit()


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
        os.system('rm -rf {}'.format(output_dir))
        os.system('mkdir {}'.format(output_dir))
        run_paths = []
        for p in paths:
            base = os.path.basename(p)
            run_path = os.path.join(output_dir, base)
            os.system('cp {} {}'.format(p, run_path))
            run_paths.append(run_path)
        write_paths(run_paths)

        for op in operations:
            all_results[test_name][op].append(
                run_standalone_trial('mp4', paths_file, op))

    print(all_results)
    return all_results


def scanner_benchmark(tests):
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
            total, prof = run_trial(dataset_name, pipeline, op,
                                    scanner_settings)
            stats = generate_statistics(prof)
            all_results[test_name][op].append((total, stats))

    print(all_results)
    return all_results


def peak_benchmark(test, frame_count, width, height):
    input_video = '/tmp/scanner_db/datasets/test/data/0_data.bin'

    def run_peak_trial(frames, op):
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
    #for test_name, paths in tests.iteritems():
    test_name = test
    if 1:
        all_results[test_name] = {}
        for op in operations:
            all_results[test_name][op] = []

        # video
        for op in operations:
            frames = frame_count
            if op == 'flow':
                frames /= 20
            all_results[test_name][op].append(
                run_peak_trial(frames, op))

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
        'io_item_size': 1024,
        'work_item_size': 128,
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
             'caffe': 10}

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
    plt.xticks(x, labels)

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


def comparison_graphs(test_name,
                      frame_counts, standalone_results, scanner_results,
                      peak_results):
    plt.clf()
    plt.title("Microbenchmarks on 1920x800 video")
    plt.ylabel("FPS")
    plt.xlabel("Pipeline")

    colors = sns.color_palette()

    x = np.arange(3)
    labels = ['caffe', 'flow', 'histogram']
    plt.xticks(x, labels)

    standalone_tests = standalone_results[test_name]
    scanner_tests = scanner_results[test_name]
    peak_tests = peak_results[test_name]
    #for test_name, tests in results.iteritems():
    if 1:
        ys = []

        standalone_y = []
        scanner_y = []
        peak_y = []
        for label in labels:
            frames = frame_counts[test_name]
            if label == 'flow':
                frames /= 20.0
            sec, timings = standalone_tests[label][0]
            if sec == -1:
                standalone_y.append(0)
            else:
                standalone_y.append(frames / sec)

            sec, timings = scanner_tests[label][0]
            if sec == -1:
                scanner_y.append(0)
            else:
                scanner_y.append(frames / sec)

            sec, timings = peak_tests[label][0]
            if sec == -1:
                peak_y.append(0)
            else:
                peak_y.append(frames / sec)

        ys.append(standalone_y)
        ys.append(scanner_y)
        ys.append(peak_y)

        print(ys)
        for (i, y) in enumerate(ys):
            xx = x+(i*0.3)
            plt.bar(xx, y, 0.3, align='center', color=colors[i])
            for xy in zip(xx, y):
                xp, yp = xy
                xp -= 0.1
                plt.annotate("{:.2f}".format(xy[1]), xy=(xp, yp))
        plt.legend(['Non-expert', 'Scanner', 'Hand-authored'],
                   loc='upper left')
        plt.tight_layout()
        plt.savefig('comparison_' + test_name + '.png', dpi=150)


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
        #'charade': ['/bigdata/wcrichto/videos/charade_short.mkv'],
        #'charade': ['/n/scanner/apoms/videos/charade_medium.mkv'],
        #'mean': ['/bigdata/wcrichto/videos/meanGirls_medium.mp4']
        #'mean': ['/n/scanner/wcrichto.new/videos/meanGirls_short.mp4'],
        'fight': ['/n/scanner/apoms/videos/fightClub_50k.mp4'],
        #'excalibur': ['/n/scanner/apoms/videos/excalibur_50k.mp4'],
    }
    frame_counts = {'charade': 21579,
                    'fight': 50350,
                    'excalibur': 50100}
    frame_wh = {'charade': {'width': 1920, 'height': 1080},
                'fight': {'width': 1920, 'height': 800},
                'excalibur': {'width': 1920, 'height': 1080}}
    t = 'fight'
    # standalone_results = standalone_benchmark(tests)
    # scanner_results = scanner_benchmark(tests)
    # peak_results = peak_benchmark(t, frame_counts[t],
    #                               frame_wh[t]['width'],
    #                               frame_wh[t]['height'])
    # print(standalone_results)
    # print(scanner_results)
    # print(peak_results)
    standalone_results = {'fight': {'caffe': [(257.01, {'load': 162.09, 'save': 0.16, 'transform': 56.68, 'eval': 94.74, 'net': 38.06, 'total': 257.01})], 'flow': [(117.21, {'load': 1.27, 'total': 117.21, 'setup': 0.23, 'save': 55.43, 'eval': 60.51})], 'histogram': [(52.57, {'load': 38.76, 'total': 52.57, 'setup': 0.14, 'save': 0.09, 'eval': 13.7})]}}
    scanner_results = {'fight': {'caffe': [(137.108575113, {'load': {'setup': '0.000395', 'task': '6.897367', 'idle': '397.431399', 'io': '6.863604'}, 'save': {'setup': '0.000009', 'task': '0.958859', 'idle': '269.800834', 'io': '0.953071'}, 'eval': {'task': '218.017667', 'evaluate': '204.490312', 'setup': '41.859860', 'evaluator_marshal': '2.993790', 'decode': '117.142565', 'idle': '280.694488', 'caffe:net': '34.965114', 'caffe:transform_input': '51.927390', 'memcpy': '2.930448'}})], 'flow': [(72.315366847, {'load': {'setup': '0.000043', 'task': '4.139968', 'idle': '137.844597', 'io': '4.114909'}, 'save': {'setup': '0.000006', 'task': '27.182298', 'idle': '111.433524', 'io': '27.181488'}, 'eval': {'task': '74.400841', 'evaluate': '73.954871', 'setup': '1.890981', 'evaluator_marshal': '0.203315', 'decode': '6.179634', 'idle': '171.577621', 'memcpy': '0.198606', 'flowcalc': '61.209670'}})], 'histogram': [(70.882922309, {'load': {'setup': '0.000010', 'task': '0.558809', 'idle': '207.476644', 'io': '0.522676'}, 'save': {'setup': '0.000007', 'task': '0.596554', 'idle': '140.104636', 'io': '0.591316'}, 'eval': {'task': '78.248330', 'evaluate': '76.713791', 'setup': '2.377652', 'histogram': '7.796299', 'decode': '68.908860', 'idle': '200.050535', 'evaluator_marshal': '1.168046', 'memcpy': '1.108454'}})]}}
    peak_results = {'fight': {'caffe': [(93.15, {'load': 0.0, 'save': 0.03, 'transform': 59.08, 'eval': 92.86, 'net': 33.78, 'total': 93.15})], 'flow': [(66.06, {'load': 0.0, 'total': 66.06, 'save': 0.0, 'eval': 65.92})], 'histogram': [(52.17, {'load': 0.0, 'total': 52.17, 'setup': 0.0, 'save': 0.0, 'eval': 12.71})]}}
    comparison_graphs(t, frame_counts, standalone_results, scanner_results,
                      peak_results)

    tests = {
        #'charade': ['/bigdata/wcrichto/videos/charade_short.mkv'],
        #'charade': ['/n/scanner/wcrichto.new/videos/charade.mkv'],
        #'mean': ['/bigdata/wcrichto/videos/meanGirls_medium.mp4']
        #'mean': ['/n/scanner/wcrichto.new/videos/meanGirls_short.mp4'],
        # 'single': ['/bigdata/wcrichto/videos/charade_short.mkv'],
        # 'many': ['/bigdata/wcrichto/videos/meanGirls_medium.mp4']
        # 'varying': ['/bigdata/wcrichto/videos/meanGirls_medium.mp4']
        'fight': ['/n/scanner/wcrichto.new/videos/movies/fightClub.mp4'],
        #'excalibur': ['/n/scanner/wrichto.new/videos/movies/excalibur.mp4'],
    }
    frame_counts = {'charade': 163430,
                    'fight': 200158,
                    'excalibur': 202275
    }
    decode_sol(tests, frame_counts)

    #kernel_sol(tests)


def bench_main(args):
    out_dir = args.output_directory
    #effective_io_rate_benchmark()
    #effective_decode_rate_benchmark()
    #dnn_rate_benchmark()
    #storage_benchmark()
    micro_comparison_driver()
    # results = standalone_benchmark()
    # standalone_graphs(results)


def graphs_main(args):
    graph_decode_rate_benchmark('decode_test.csv')


def trace_main(args):
    dataset = args.dataset
    job = args.job
    db = scanner.Scanner()
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
