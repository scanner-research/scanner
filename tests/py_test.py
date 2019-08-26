import scannerpy
from scannerpy import (Client, Config, DeviceType, FrameType, Job,
                       ScannerException, Kernel, protobufs, NullElement, SliceList, CacheMode,
                       PerfParams)
from scannerpy.storage import NamedStream, NamedVideoStream
from typing import Dict, List, Sequence, Tuple, Any
import tempfile
import toml
import pytest
from subprocess import check_call as run
from multiprocessing import Process, Queue
import requests
import imp
import os.path
import socket
import numpy as np
import sys
import grpc
import struct
import pickle
import subprocess
import json
import time
import cv2
from scannerpy import types

cwd = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    try:
        run(['nvidia-smi'])
        has_gpu = True
    except (OSError, subprocess.CalledProcessError) as e:
        has_gpu = False

    gpu = pytest.mark.skipif(not has_gpu, reason='need GPU to run')
    slow = pytest.mark.skipif(
        not pytest.config.getoption('--runslow'),
        reason='need --runslow option to run')

else:
    gpu = pytest.mark.skipif(True, reason='')
    slow = pytest.mark.skipif(True, reason='')


@slow
def test_examples():
    def run_py(arg):
        [d, f] = arg
        print(f)
        run('cd {}/../examples/{} && python3 {}.py'.format(cwd, d, f),
            shell=True)

    examples = [['face_detection', 'face_detect'],
                ['shot_detection', 'shot_detect']]

    for e in examples:
        run_py(e)


def make_config(master_port=None, worker_port=None, path=None):
    cfg = Config.default_config()
    cfg['network']['master'] = 'localhost'
    cfg['storage']['db_path'] = tempfile.mkdtemp()
    if master_port is not None:
        cfg['network']['master_port'] = master_port
    if worker_port is not None:
        cfg['network']['worker_port'] = worker_port

    if path is not None:
        with open(path, 'w') as f:
            cfg_path = path
            f.write(toml.dumps(cfg))
    else:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cfg_path = f.name
            f.write(bytes(toml.dumps(cfg), 'utf-8'))
    return (cfg_path, cfg)


def download_videos():
    # Download video from GCS
    url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        host = socket.gethostname()
        # HACK: special proxy case for Ocean cluster
        if host in ['ocean', 'crissy', 'pismo', 'stinson']:
            resp = requests.get(
                url,
                stream=True,
                proxies={'https': 'http://proxy.pdl.cmu.edu:3128/'})
        else:
            resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        vid1_path = f.name

    # Make a second one shorter than the first
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        vid2_path = f.name
    run([
        'ffmpeg', '-y', '-i', vid1_path, '-ss', '00:00:00', '-t', '00:00:10',
        '-c:v', 'libx264', '-strict', '-2', vid2_path
    ])

    return (vid1_path, vid2_path)

@pytest.fixture(scope="module")
def cl():
    # Create new config
    (cfg_path, cfg) = make_config()

    # Setup and ingest video
    with Client(config_path=cfg_path, debug=True) as cl:
        (vid1_path, vid2_path) = download_videos()

        cl.load_op(
            os.path.abspath(os.path.join(cwd, '..', 'build/tests/libscanner_tests.so')),
            os.path.abspath(os.path.join(cwd, '..', 'build/tests/test_ops_pb2.py')))

        cl.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        cl.ingest_videos(
            [('test1_inplace', vid1_path), ('test2_inplace', vid2_path)],
            inplace=True)

        yield cl

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_new_client(cl):
    pass


def test_perf_params(cl):
    frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
    hist = cl.ops.Histogram(frame=frame)
    ghist = cl.streams.Gather(hist, [[0]])
    output_op = cl.io.Output(ghist, [NamedStream(cl, '_ignore')])

    cl.run(output_op, PerfParams.manual(10, 10),
           show_progress=False, cache_mode=CacheMode.Overwrite)

    cl.run(output_op, PerfParams.estimate(),
           show_progress=False, cache_mode=CacheMode.Overwrite)


def test_auto_ingest(cl):
    (vid1_path, vid2_path) = download_videos()
    input = NamedVideoStream(cl, 'test3', path=vid1_path)
    frame = cl.io.Input([input])
    hist = cl.ops.Histogram(frame=frame)
    output = NamedStream(cl, 'test_hist')
    output_op = cl.io.Output(hist, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    run(['rm', '-rf', vid1_path, vid2_path])


def test_table_properties(cl):
    for name, i in [('test1', 0), ('test1_inplace', 2)]:
        table = cl.table(name)
        assert table.id() == i
        assert table.name() == name
        assert table.num_rows() == 720
        assert [c for c in table.column_names()] == ['index', 'frame']


def test_summarize(cl):
    cl.summarize()


def test_load_video_column(cl):
    for name in ['test1', 'test1_inplace']:
        next(NamedVideoStream(cl, name).load())


def test_gather_video_column(cl):
    for name in ['test1', 'test1_inplace']:
        # Gather rows
        rows = [0, 10, 100, 200]
        frames = list(NamedVideoStream(cl, name).load(rows=rows))
        assert len(frames) == len(rows)


def test_profiler(cl):
    frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
    hist = cl.ops.Histogram(frame=frame)
    ghist = cl.streams.Gather(hist, [[0]])
    output_op = cl.io.Output(ghist, [NamedStream(cl, '_ignore')])

    time_start = time.time()
    job_id = cl.run(output_op, PerfParams.estimate(), show_progress=False, cache_mode=CacheMode.Overwrite)
    print('Time', time.time() - time_start)
    profile = cl.get_profile(job_id)
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.trace')
    f.close()
    profile.write_trace(f.name)
    profile.statistics()
    run(['rm', '-f', f.name])


def test_new_table(cl):
    def b(s):
        return bytes(s, 'utf-8')

    cl.new_table('test', ['col1', 'col2'],
                 [[b('r00'), b('r01')], [b('r10'), b('r11')]])
    t = cl.table('test')
    assert (t.num_rows() == 2)
    assert (next(t.column('col2').load()) == b('r01'))


def test_multiple_outputs(cl):
    sampler = cl.streams.Range
    def run_job(args_1, args_2):
        frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
        sample_frame_1 = cl.streams.Range(input=frame, ranges=[args_1])
        sample_frame_2 = cl.streams.Range(input=frame, ranges=[args_2])
        output_op_1 = cl.io.Output(sample_frame_1, [NamedVideoStream(cl, 'test_mp_1')])
        output_op_2 = cl.io.Output(sample_frame_2, [NamedVideoStream(cl, 'test_mp_2')])

        cl.run([output_op_1, output_op_2], PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    # This should fail
    sampler_args_1 = {'start': 0, 'end': 30}
    sampler_args_2 = {'start': 0, 'end': 15}
    exc = False
    try:
        run_job(sampler_args_1, sampler_args_2)
    except ScannerException:
        exc = True

    assert exc

    # This should succeed
    sampler_args_1 = {'start': 0, 'end': 30}
    expected_rows_1 = 30
    sampler_args_2 = {'start': 30, 'end': 60}
    expected_rows_2 = 30

    run_job(sampler_args_1, sampler_args_2)

    num_rows = 0
    for _ in cl.table('test_mp_1').column('frame').load():
        num_rows += 1
    assert num_rows == expected_rows_1

    num_rows = 0
    for _ in cl.table('test_mp_2').column('frame').load():
        num_rows += 1
    assert num_rows == expected_rows_2

    # This should succeed
    frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
    sample_frame_1 = cl.streams.Range(input=frame, ranges=[sampler_args_1])
    output_op_1 = cl.io.Output(sample_frame_1, [NamedVideoStream(cl, 'test_mp_1')])
    output_op_2 = cl.io.Output(sample_frame_1, [NamedVideoStream(cl, 'test_mp_2')])

    cl.run([output_op_1, output_op_2], PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    for _ in cl.table('test_mp_1').column('frame').load():
        num_rows += 1
    assert num_rows == expected_rows_1


def test_sample(cl):
    def run_sampler_job(sampler, sampler_args, expected_rows):
        frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
        sample_frame = sampler(input=frame, **sampler_args)
        output = NamedVideoStream(cl, 'test_sample')
        output_op = cl.io.Output(sample_frame, [output])
        cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

        num_rows = len(list(output.load()))
        assert num_rows == expected_rows

    # Stride
    expected = int((cl.table('test1').num_rows() + 8 - 1) / 8)
    run_sampler_job(cl.streams.Stride, {'strides': [{'stride': 8}]}, expected)
    # Range
    run_sampler_job(cl.streams.Range, {'ranges': [{'start': 0, 'end': 30}]}, 30)
    # Strided Range
    run_sampler_job(cl.streams.StridedRange, {'ranges': [{
        'start': 0,
        'end': 300,
        'stride': 10
    }]}, 30)
    # Gather
    run_sampler_job(cl.streams.Gather, {'indices': [[0, 150, 377, 500]]}, 4)


def test_space(cl):
    def run_spacer_job(spacer, spacing):
        frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
        hist = cl.ops.Histogram(frame=frame)
        space_hist = spacer(input=hist, spacings=[spacing])
        output = NamedStream(cl, 'test_space')
        output_op = cl.io.Output(space_hist, [output])
        cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
        return output

    # # Repeat
    spacing_distance = 8
    table = run_spacer_job(cl.streams.Repeat, spacing_distance)
    num_rows = 0
    for hist in table.load():
        # Verify outputs are repeated correctly
        if num_rows % spacing_distance == 0:
            ref_hist = hist
        assert len(hist) == 3
        for c in range(len(hist)):
            assert (ref_hist[c] == hist[c]).all()
        num_rows += 1
    assert num_rows == NamedVideoStream(cl, 'test1').len() * spacing_distance

    # Null
    table = run_spacer_job(cl.streams.RepeatNull, spacing_distance)
    num_rows = 0
    for hist in table.load():
        # Verify outputs are None for null rows
        if num_rows % spacing_distance == 0:
            assert not isinstance(hist, NullElement)
            assert len(hist) == 3
            assert hist[0].shape[0] == 16
        else:
            assert isinstance(hist, NullElement)
        num_rows += 1
    assert num_rows == NamedVideoStream(cl, 'test1').len() * spacing_distance


def test_stream_args(cl):
    frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
    resized_frame = cl.ops.Resize(frame=frame, width=[640], height=[480])
    range_frame = cl.streams.Range(resized_frame, [(0, 10)])
    output_stream = NamedVideoStream(cl, 'test_stream_args')
    output_op = cl.io.Output(range_frame, [output_stream])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    list(output_stream.load())


def test_slice(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    slice_frame = cl.streams.Slice(frame, partitions=[cl.partitioner.all(50)])
    unsliced_frame = cl.streams.Unslice(slice_frame)
    output = NamedStream(cl, 'test_slicing')
    output_op = cl.io.Output(unsliced_frame, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    assert input.len() == output.len()


def test_overlapping_slice(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    slice_frame = cl.streams.Slice(frame, partitions=[
        cl.partitioner.strided_ranges([(0, 15), (5, 25), (15, 35)], 1)])
    sample_frame = cl.streams.Range(slice_frame, ranges=[SliceList([
        {'start': 0, 'end': 10},
        {'start': 5, 'end': 15},
        {'start': 5, 'end': 15},
    ])])
    unsliced_frame = cl.streams.Unslice(sample_frame)
    output = NamedStream(cl, 'test_slicing')
    output_op = cl.io.Output(unsliced_frame, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    assert output.len() == 30


@scannerpy.register_python_op()
class TestSliceArgs(Kernel):
    def __init__(self, config):
        pass

    def close(self):
        pass

    def new_stream(self, arg):
        self.arg = arg

    def execute(self, frame: FrameType) -> Any:
        return self.arg


def test_slice_args(cl):
    frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
    slice_frame = cl.streams.Slice(frame, [cl.partitioner.ranges(
        [[0, 1], [1, 2], [2, 3]])])
    test = cl.ops.TestSliceArgs(frame=slice_frame, arg=[SliceList([i for i in range(3)])])
    unsliced_frame = cl.streams.Unslice(test)
    output = NamedStream(cl, 'test_slicing')
    output_op = cl.io.Output(unsliced_frame, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    list(output.load())


def test_bounded_state(cl):
    warmup = 3

    frame = cl.io.Input([NamedVideoStream(cl, 'test1')])
    increment = cl.ops.TestIncrementBounded(ignore=frame, bounded_state=warmup)
    sampled_increment = cl.streams.Gather(increment, indices=[[0, 10, 25, 26, 27]])
    output = NamedStream(cl, 'test_bounded_state')
    output_op = cl.io.Output(sampled_increment, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    expected_output = [0, warmup, warmup, warmup + 1, warmup + 2]
    for buf in output.load():
        (val, ) = struct.unpack('=q', buf)
        assert val == expected_output[num_rows]
        num_rows += 1
    assert num_rows == 5


def test_unbounded_state(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    slice_frame = cl.streams.Slice(frame, partitions=[cl.partitioner.all(50)])
    increment = cl.ops.TestIncrementUnbounded(ignore=slice_frame)
    unsliced_increment = cl.streams.Unslice(increment)
    output = NamedStream(cl, 'test_unbounded_state')
    output_op = cl.io.Output(unsliced_increment, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    assert output.len() == input.len()


class DeviceTestBench:
    def test_cpu(self, cl):
        self.run(cl, DeviceType.CPU)

    @gpu
    def test_gpu(self, cl):
        self.run(cl, DeviceType.GPU)


class TestInplace(DeviceTestBench):
    def run(self, cl, device):
        input = NamedVideoStream(cl, 'test1_inplace')
        frame = cl.io.Input([input])
        hist = cl.ops.Histogram(frame=frame, device=device)
        output = NamedStream(cl, 'test_hist')
        output_op = cl.io.Output(hist, [output])

        cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
        next(output.load())


def test_stencil(cl):
    input = NamedVideoStream(cl, 'test1')

    frame = cl.io.Input([input])
    sample_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    flow = cl.ops.OpticalFlow(frame=sample_frame, stencil=[-1, 0])
    output = NamedStream(cl, 'test_stencil')
    output_op = cl.io.Output(flow, [output])
    cl.run(output_op,
           PerfParams.estimate(pipeline_instances_per_node=1),
           cache_mode=CacheMode.Overwrite,
           show_progress=False)
    assert output.len() == 1

    frame = cl.io.Input([input])
    sample_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    flow = cl.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output = NamedStream(cl, 'test_stencil')
    output_op = cl.io.Output(flow, [output])
    cl.run(output_op,
           PerfParams.estimate(pipeline_instances_per_node=1),
           cache_mode=CacheMode.Overwrite,
           show_progress=False)

    frame = cl.io.Input([input])
    sample_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 2}])
    flow = cl.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output = NamedStream(cl, 'test_stencil')
    output_op = cl.io.Output(flow, [output])
    cl.run(output_op,
           PerfParams.estimate(pipeline_instances_per_node=1),
           cache_mode=CacheMode.Overwrite,
           show_progress=False)
    assert output.len() == 2

    frame = cl.io.Input([input])
    flow = cl.ops.OpticalFlow(frame=frame, stencil=[-1, 0])
    sample_flow = cl.streams.Range(flow, ranges=[{'start': 0, 'end': 1}])
    output = NamedStream(cl, 'test_stencil')
    output_op = cl.io.Output(sample_flow, [output])
    cl.run(output_op,
           PerfParams.estimate(pipeline_instances_per_node=1),
           cache_mode=CacheMode.Overwrite,
           show_progress=False)
    assert output.len() == 1


def test_wider_than_packet_stencil(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    sample_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 3}])
    flow = cl.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output = NamedStream(cl, 'test_stencil')
    output_op = cl.io.Output(flow, [output])

    cl.run(
        output_op,
        PerfParams.manual(1, 1, pipeline_instances_per_node=1),
        cache_mode=CacheMode.Overwrite,
        show_progress=False)

    assert output.len() == 3


# def test_packed_file_source(cl):
#     # Write test file
#     path = '/tmp/cpp_source_test'
#     with open(path, 'wb') as f:
#         num_elements = 4
#         f.write(struct.pack('=Q', num_elements))
#         # Write sizes
#         for i in range(num_elements):
#             f.write(struct.pack('=Q', 8))
#         # Write data
#         for i in range(num_elements):
#             f.write(struct.pack('=Q', i))

#     data = cl.sources.PackedFile()
#     pass_data = cl.ops.Pass(input=data)
#     output_op = cl.sinks.Column(columns={'integer': pass_data})
#     job = Job(op_args={
#         data: {
#             'path': path
#         },
#         output_op: 'test_cpp_source',
#     })

#     cl.run(output_op, [job], PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
#     tables = [cl.table('test_cpp_source')]

#     num_rows = 0
#     for buf in tables[0].column('integer').load():
#         (val, ) = struct.unpack('=Q', buf)
#         assert val == num_rows
#         num_rows += 1
#     assert num_elements == tables[0].num_rows()


@scannerpy.register_python_op()
class TestPy(Kernel):
    def __init__(self, config, kernel_arg):
        assert (kernel_arg == 1)
        self.x = 20
        self.y = 20

    def new_stream(self, x, y):
        self.x = x
        self.y = y

    def execute(self, frame: FrameType) -> Any:
        point = {}
        point['x'] = self.x
        point['y'] = self.y
        return point


def test_python_kernel(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 3}])
    test_out = cl.ops.TestPy(frame=range_frame, kernel_arg=1, x=[0], y=[0])
    output = NamedStream(cl, 'test_hist')
    output_op = cl.io.Output(test_out, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


@scannerpy.register_python_op()
class ResourceTest(Kernel):
    def __init__(self, config, path):
        self.path = path

    def fetch_resources(self):
        with open(self.path, 'r') as f:
            n = int(f.read())

        with open(self.path, 'w') as f:
            f.write(str(n + 1))

    def setup_with_resources(self):
        with open(self.path, 'r') as f:
            assert int(f.read()) == 1

    def execute(self, frame: FrameType) -> Any:
        return None


def test_fetch_resources(cl):
    with tempfile.NamedTemporaryFile() as f:
        f.write(b'0')
        f.flush()

        input = NamedVideoStream(cl, 'test1')
        frame = cl.io.Input([input])
        range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 3}])
        test_out = cl.ops.ResourceTest(frame=frame, path=f.name)
        output = NamedStream(cl, 'test_hist')
        output_op = cl.io.Output(test_out, [output])
        cl.run(
            output_op, PerfParams.estimate(pipeline_instances_per_node=2),
            cache_mode=CacheMode.Overwrite, show_progress=False)


@scannerpy.register_python_op(batch=50)
class TestPyBatch(Kernel):
    def execute(self, frame: Sequence[FrameType]) -> Sequence[bytes]:
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        input_count = len(frame)
        column_count = 1
        return [point.SerializeToString() for _ in range(input_count)]


def test_python_batch_kernel(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    test_out = cl.ops.TestPyBatch(frame=range_frame, batch=50)
    output = NamedStream(cl, 'test_hist')
    output_op = cl.io.Output(test_out, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


@scannerpy.register_python_op(stencil=[0, 1])
class TestPyStencil(Kernel):
    def execute(self, frame: Sequence[FrameType]) -> bytes:
        assert len(frame) == 2
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        return point.SerializeToString()


def test_python_stencil_kernel(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    test_out = cl.ops.TestPyStencil(frame=range_frame)
    output = NamedStream(cl, 'test_hist')
    output_op = cl.io.Output(test_out, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


@scannerpy.register_python_op(stencil=[0, 1], batch=50)
class TestPyStencilBatch(Kernel):
    def __init__(self, config):
        pass

    def close(self):
        pass

    def execute(self, frame: Sequence[Sequence[FrameType]]) -> Sequence[bytes]:

        assert len(frame[0]) == 2
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        input_count = len(frame)
        column_count = 1
        return [point.SerializeToString() for _ in range(input_count)]


def test_python_stencil_batch_kernel(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    test_out = cl.ops.TestPyStencilBatch(frame=range_frame, batch=50)
    output = NamedStream(cl, 'test_hist')
    output_op = cl.io.Output(test_out, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_bind_op_args(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input, input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 1} for _ in range(2)])
    test_out = cl.ops.TestPy(frame=range_frame, kernel_arg=1, x=[1, 10], y=[5, 50])
    outputs = [NamedStream(cl, 'test_hist_0'), NamedStream(cl, 'test_hist_1')]
    output_op = cl.io.Output(test_out, outputs)
    pairs = [(1, 5), (10, 50)]
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    for i, (x, y) in enumerate(pairs):
        values = list(outputs[i].load())
        p = values[0]
        assert p['x'] == x
        assert p['y'] == y


@scannerpy.register_python_op()
class TestPyVariadic(Kernel):
    def execute(self, *frame: Tuple[FrameType, ...]) -> FrameType:
        assert len(frame) == 3
        return frame[0]


def test_py_variadic(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    out_frame = cl.ops.TestPyVariadic(range_frame, range_frame, range_frame)
    output = NamedVideoStream(cl, 'test_variadic')
    output_op = cl.io.Output(out_frame.lossless(), [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_lossless(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = cl.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = NamedVideoStream(cl, 'test_blur')
    output_op = cl.io.Output(blurred_frame.lossless(), [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_compress(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = cl.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = NamedVideoStream(cl, 'test_blur')
    output_op = cl.io.Output(blurred_frame.compress('video', bitrate=1 * 1024 * 1024), [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_save_mp4(cl):
    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = cl.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = NamedVideoStream(cl, 'test_save_mp4')
    output_op = cl.io.Output(blurred_frame, [output])
    cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite, show_progress=False)

    f = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    f.close()
    output.save_mp4(f.name)
    run(['rm', '-rf', f.name])


@pytest.fixture()
def no_workers_cl():
    # Create new config
    (cfg_path, cfg) = make_config(master_port='5020', worker_port='5021')

    # Setup and ingest video
    with Client(workers=[], config_path=cfg_path, enable_watchdog=False,
                  debug=True) as cl:
        (vid1_path, vid2_path) = download_videos()

        cl.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield cl

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_no_workers(no_workers_cl):
    cl = no_workers_cl

    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    hist = cl.ops.Histogram(frame=frame)
    output_op = cl.io.Output(hist, [NamedStream(cl, '_ignore')])

    exc = False
    try:
        cl.run(output_op, PerfParams.estimate(), show_progress=False, cache_mode=CacheMode.Overwrite)
    except ScannerException:
        exc = True

    assert exc


@pytest.fixture()
def fault_cl():
    # Create new config
    (cfg_path, cfg) = make_config(
        master_port='5010', worker_port='5011', path='/tmp/config_test')

    # Setup and ingest video
    with Client(
            master='localhost:5010',
            workers=[],
            config_path=cfg_path,
            no_workers_timeout=120,
            enable_watchdog=False,
            debug=True) as cl:
        (vid1_path, vid2_path) = download_videos()

        cl.load_op(
            os.path.abspath(os.path.join(cwd, '..', 'build/tests/libscanner_tests.so')),
            os.path.abspath(os.path.join(cwd, '..', 'build/tests/test_ops_pb2.py')))

        cl.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield cl

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


# def test_clean_worker_shutdown(fault_cl):
#     spawn_port = 5010
#     def worker_shutdown_task(config, master_address, worker_address):
#         from scannerpy import ProtobufGenerator, Config, start_worker
#         import time
#         import grpc
#         import subprocess

#         c = Config(None)

#         import scanner.metadata_pb2 as metadata_types
#         import scanner.engine.rpc_pb2 as rpc_types
#         import scanner.types_pb2 as misc_types
#         import libscanner as bindings

#         protobufs = ProtobufGenerator(config)

#         # Wait to kill worker
#         time.sleep(8)
#         # Kill worker
#         channel = grpc.insecure_channel(
#             worker_address,
#             options=[('grpc.max_message_length', 24499183 * 2)])
#         worker = protobufs.WorkerStub(channel)

#         try:
#             worker.Shutdown(protobufs.Empty())
#         except grpc.RpcError as e:
#             status = e.code()
#             if status == grpc.StatusCode.UNAVAILABLE:
#                 print('could not shutdown worker!')
#                 exit(1)
#             else:
#                 raise ScannerException('Worker errored with status: {}'
#                                        .format(status))

#         # Wait a bit
#         time.sleep(15)
#         script_dir = os.path.dirname(os.path.realpath(__file__))
#         subprocess.call(['python ' +  script_dir + '/spawn_worker.py'],
#                         shell=True)

#     master_addr = fault_cl._master_address
#     worker_addr = fault_cl._worker_addresses[0]
#     shutdown_process = Process(target=worker_shutdown_task,
#                              args=(fault_cl.config, master_addr, worker_addr))
#     shutdown_process.daemon = True
#     shutdown_process.start()

#     frame = fault_cl.sources.FrameColumn()
#     range_frame = frame.sample()
#     sleep_frame = fault_cl.ops.SleepFrame(ignore = range_frame)
#     output_op = fault_cl.sinks.Column(columns=[sleep_frame])

#     job = Job(
#         op_args={
#             frame: fault_cl.table('test1').column('frame'),
#             range_frame: fault_cl.sampler.range(0, 15),
#             output_op: 'test_shutdown',
#         }
#     )
#
#     table = fault_cl.run(output_op, [job], pipeline_instances_per_node=1, PerfParams.estimate(), cache_mode=CacheMode.Overwrite,
#                          show_progress=False)
#     table = table[0]
#     assert len([_ for _, _ in table.column('dummy').load()]) == 15

#     # Shutdown the spawned worker
#     channel = grpc.insecure_channel(
#         'localhost:' + str(spawn_port),
#         options=[('grpc.max_message_length', 24499183 * 2)])
#     worker = fault_cl.protobufs.WorkerStub(channel)

#     try:
#         worker.Shutdown(fault_cl.protobufs.Empty())
#     except grpc.RpcError as e:
#         status = e.code()
#         if status == grpc.StatusCode.UNAVAILABLE:
#             print('could not shutdown worker!')
#             exit(1)
#         else:
#             raise ScannerException('Worker errored with status: {}'
#                                    .format(status))
#     shutdown_process.join()


def test_fault_tolerance(fault_cl):
    force_kill_spawn_port = 5012
    normal_spawn_port = 5013

    def worker_killer_task(config, master_address):
        from scannerpy import Config, start_worker, protobufs
        import time
        import grpc
        import subprocess
        import signal
        import os

        import scanner.metadata_pb2 as metadata_types
        import scanner.engine.rpc_pb2 as rpc_types
        import scanner.types_pb2 as misc_types


        # Spawn a worker that we will force kill
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.devnull, 'w') as fp:
            p = subprocess.Popen(
                [
                    'python3 ' + script_dir +
                    '/spawn_worker.py {:d}'.format(force_kill_spawn_port)
                ],
                shell=True,
                stdout=fp,
                stderr=fp,
                preexec_fn=os.setsid)

            # Wait a bit for the worker to do its thing
            time.sleep(10)

            # Force kill worker process to trigger fault tolerance
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.kill()
            p.communicate()

            # Wait for fault tolerance to kick in
            time.sleep(15)

            # Spawn the worker again
            subprocess.call(
                [
                    'python3 ' + script_dir +
                    '/spawn_worker.py {:d}'.format(normal_spawn_port)
                ],
                shell=True)

    master_addr = fault_cl._master_address
    killer_process = Process(
        target=worker_killer_task, args=(fault_cl.config, master_addr))
    killer_process.daemon = True
    killer_process.start()

    input = NamedVideoStream(fault_cl, 'test1')
    frame = fault_cl.io.Input([input])
    range_frame = fault_cl.streams.Range(frame, ranges=[{'start': 0, 'end': 20}])
    sleep_frame = fault_cl.ops.SleepFrame(ignore=range_frame)
    output = NamedStream(fault_cl, 'test_fault')
    output_op = fault_cl.io.Output(sleep_frame, [output])

    fault_cl.run(
        output_op,
        PerfParams.estimate(pipeline_instances_per_node=1),
        cache_mode=CacheMode.Overwrite,
        show_progress=False)

    assert output.len() == 20

    # Shutdown the spawned worker
    channel = grpc.insecure_channel(
        'localhost:' + str(normal_spawn_port),
        options=[('grpc.max_message_length', 24499183 * 2)])
    worker = protobufs.WorkerStub(channel)

    try:
        worker.Shutdown(protobufs.Empty())
    except grpc.RpcError as e:
        status = e.code()
        if status == grpc.StatusCode.UNAVAILABLE:
            print('could not shutdown worker!')
            exit(1)
        else:
            raise ScannerException(
                'Worker errored with status: {}'.format(status))
    killer_process.join()


@pytest.fixture()
def blacklist_cl():
    # Create new config
    (cfg_path, cfg) = make_config(master_port='5055', worker_port='5060')

    # Setup and ingest video
    master = 'localhost:5055'
    workers = ['localhost:{:04d}'.format(5060 + d) for d in range(4)]
    with Client(
            config_path=cfg_path,
            no_workers_timeout=120,
            master=master,
            workers=workers,
            enable_watchdog=False) as cl:
        (vid1_path, vid2_path) = download_videos()

        cl.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield cl

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_job_blacklist(blacklist_cl):
    # NOTE(wcrichto): this class must NOT be at the top level. If it is, then pytest injects
    # some of its dependencies, and sending this class to an external Scanner process will fail
    # with a missing "py_test" import..
    @scannerpy.register_python_op()
    class TestPyFail(Kernel):
        def execute(self, frame: FrameType) -> bytes:
            raise ScannerException('Test')

    cl = blacklist_cl

    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    failed_output = cl.ops.TestPyFail(frame=range_frame)
    output = NamedVideoStream(cl, 'test_py_fail')
    output_op = cl.io.Output(failed_output, [output])
    cl.run(
        output_op,
        PerfParams.estimate(pipeline_instances_per_node=1),
        cache_mode=CacheMode.Overwrite,
        show_progress=False)
    assert not output.committed()


@pytest.fixture()
def timeout_cl():
    # Create new config
    (cfg_path, cfg) = make_config(master_port='5155', worker_port='5160')

    # Setup and ingest video
    master = 'localhost:5155'
    workers = ['localhost:{:04d}'.format(5160 + d) for d in range(4)]
    with Client(
            config_path=cfg_path,
            no_workers_timeout=120,
            master=master,
            workers=workers,
            enable_watchdog=False) as cl:
        (vid1_path, vid2_path) = download_videos()

        cl.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield cl

        for worker in workers:
            channel = grpc.insecure_channel(worker)
            worker_stub = protobufs.WorkerStub(channel)
            try:
                worker_stub.Shutdown(
                    protobufs.Empty(), timeout=cl._grpc_timeout)
            except grpc.RpcError as e:
                pass

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_job_timeout(timeout_cl):
    @scannerpy.register_python_op()
    def timeout_fn(self, frame: FrameType) -> bytes:
        time.sleep(5)
        return bytes('what', 'utf-8')

    cl = timeout_cl

    input = NamedVideoStream(cl, 'test1')
    frame = cl.io.Input([input])
    range_frame = cl.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    sleep_frame = cl.ops.timeout_fn(frame=range_frame)
    output = NamedVideoStream(cl, 'test_timeout')
    output_op = cl.io.Output(sleep_frame, [output])

    cl.run(
        output_op,
        PerfParams.estimate(pipeline_instances_per_node=1),
        task_timeout=0.1,
        cache_mode=CacheMode.Overwrite,
        show_progress=False)

    assert not output.committed()



@scannerpy.register_python_op(name='CacheTest')
def cache_test(config, n: Any) -> Any:
    return n + 1


# def test_cache_mode(cl):
#     cl.new_table('test_cache_input', ['column'], [[pickle.dumps(0)]])
#     cl.new_table('test_cache_input2', ['column'], [[pickle.dumps(1)]])

#     n = cl.io.Input([NamedStream(cl, 'test_cache_input')])
#     out = cl.ops.CacheTest(n=n)
#     output = NamedStream(cl, 'test_cache')
#     output_op = cl.io.Output(out, [output])
#     cl.run(output_op, PerfParams.estimate())
# ge
#     assert next(output.load()) == 1

#     exc = False
#     try:
#         cl.run(output_op, PerfParams.estimate())
#     except ScannerException:
#         exc = True
#     assert exc

#     n = cl.io.Input([NamedStream(cl, 'test_cache_input2')])
#     out = cl.ops.CacheTest(n=n)
#     output_op = cl.io.Output(out, [output])

#     cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Ignore)
#     assert next(output.load()) == 1

#     cl.run(output_op, PerfParams.estimate(), cache_mode=CacheMode.Overwrite)
#     assert next(output.load()) == 2


def test_tutorial():
    def run_py(path):
        print(path)
        run('cd {}/../examples/tutorials && python3 {}.py'.format(cwd, path),
            shell=True)

    run('cd {}/../examples/tutorials/resize_op && '
        'mkdir -p build && cd build && cmake -D SCANNER_PATH={} .. && '
        'make'.format(cwd, cwd + '/..'),
        shell=True)

    tutorials = [
        '00_basic', '01_defining_python_ops', '02_op_attributes', '03_sampling', '04_slicing',
        '05_sources_sinks', '06_compression', '07_profiling', '08_defining_cpp_ops'
    ]

    for t in tutorials:
        run_py(t)
