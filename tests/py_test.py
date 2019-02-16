import scannerpy
from scannerpy import (Database, Config, DeviceType, FrameType, Job,
                       ScannerException, Kernel, protobufs, NullElement, SliceList, CacheMode)
from scannerpy.storage import (
    ScannerStream, ScannerFrameStream, FilesStream, PythonStream, SQLInputStream,
    SQLOutputStream, SQLStorage, AudioStream, CaptionStream)
from scannerpy.stdlib import readers
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
from testing.postgresql import Postgresql
import psycopg2
import json
import time

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
def db():
    # Create new config
    (cfg_path, cfg) = make_config()

    # Setup and ingest video
    with Database(config_path=cfg_path, debug=True) as db:
        (vid1_path, vid2_path) = download_videos()

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        db.ingest_videos(
            [('test1_inplace', vid1_path), ('test2_inplace', vid2_path)],
            inplace=True)

        yield db

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_new_database(db):
    pass


def test_table_properties(db):
    for name, i in [('test1', 0), ('test1_inplace', 2)]:
        table = db.table(name)
        assert table.id() == i
        assert table.name() == name
        assert table.num_rows() == 720
        assert [c for c in table.column_names()] == ['index', 'frame']


def test_summarize(db):
    db.summarize()


def test_load_video_column(db):
    for name in ['test1', 'test1_inplace']:
        next(ScannerFrameStream(db, name).load())


def test_gather_video_column(db):
    for name in ['test1', 'test1_inplace']:
        # Gather rows
        rows = [0, 10, 100, 200]
        frames = list(ScannerFrameStream(db, name).load(rows=rows))
        assert len(frames) == len(rows)


def test_profiler(db):
    frame = db.io.Input([ScannerFrameStream(db, 'test1')])
    hist = db.ops.Histogram(frame=frame)
    ghist = db.streams.Gather(hist, [[0]])
    output_op = db.io.Output(ghist, [ScannerStream(db, '_ignore')])

    time_start = time.time()
    db.run(output_op, show_progress=False, cache_mode=CacheMode.Overwrite)
    output = [db.table('_ignore')]
    print('Time', time.time() - time_start)
    profiler = output[0].profiler()
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.trace')
    f.close()
    profiler.write_trace(f.name)
    profiler.statistics()
    run(['rm', '-f', f.name])


def test_new_table(db):
    def b(s):
        return bytes(s, 'utf-8')

    db.new_table('test', ['col1', 'col2'],
                 [[b('r00'), b('r01')], [b('r10'), b('r11')]])
    t = db.table('test')
    assert (t.num_rows() == 2)
    assert (next(t.column('col2').load()) == b('r01'))


def test_multiple_outputs(db):
    sampler = db.streams.Range
    def run_job(args_1, args_2):
        frame = db.io.Input([ScannerFrameStream(db, 'test1')])
        sample_frame_1 = db.streams.Range(input=frame, ranges=[args_1])
        sample_frame_2 = db.streams.Range(input=frame, ranges=[args_2])
        output_op_1 = db.io.Output(sample_frame_1, [ScannerFrameStream(db, 'test_mp_1')])
        output_op_2 = db.io.Output(sample_frame_2, [ScannerFrameStream(db, 'test_mp_2')])

        db.run([output_op_1, output_op_2], cache_mode=CacheMode.Overwrite, show_progress=False)

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
    for _ in db.table('test_mp_1').column('frame').load():
        num_rows += 1
    assert num_rows == expected_rows_1

    num_rows = 0
    for _ in db.table('test_mp_2').column('frame').load():
        num_rows += 1
    assert num_rows == expected_rows_2

    # This should succeed
    frame = db.io.Input([ScannerFrameStream(db, 'test1')])
    sample_frame_1 = db.streams.Range(input=frame, ranges=[sampler_args_1])
    output_op_1 = db.io.Output(sample_frame_1, [ScannerFrameStream(db, 'test_mp_1')])
    output_op_2 = db.io.Output(sample_frame_1, [ScannerFrameStream(db, 'test_mp_2')])

    db.run([output_op_1, output_op_2], cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    for _ in db.table('test_mp_1').column('frame').load():
        num_rows += 1
    assert num_rows == expected_rows_1


def test_sample(db):
    def run_sampler_job(sampler, sampler_args, expected_rows):
        frame = db.io.Input([ScannerFrameStream(db, 'test1')])
        sample_frame = sampler(input=frame, **sampler_args)
        output = ScannerFrameStream(db, 'test_sample')
        output_op = db.io.Output(sample_frame, [output])
        db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

        num_rows = len(list(output.load()))
        assert num_rows == expected_rows

    # Stride
    expected = int((db.table('test1').num_rows() + 8 - 1) / 8)
    run_sampler_job(db.streams.Stride, {'strides': [{'stride': 8}]}, expected)
    # Range
    run_sampler_job(db.streams.Range, {'ranges': [{'start': 0, 'end': 30}]}, 30)
    # Strided Range
    run_sampler_job(db.streams.StridedRange, {'ranges': [{
        'start': 0,
        'end': 300,
        'stride': 10
    }]}, 30)
    # Gather
    run_sampler_job(db.streams.Gather, {'indices': [[0, 150, 377, 500]]}, 4)


def test_space(db):
    def run_spacer_job(spacer, spacing):
        frame = db.io.Input([ScannerFrameStream(db, 'test1')])
        hist = db.ops.Histogram(frame=frame)
        space_hist = spacer(input=hist, spacings=[spacing])
        output = ScannerStream(db, 'test_space')
        output_op = db.io.Output(space_hist, [output])
        db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
        return output

    # # Repeat
    spacing_distance = 8
    table = run_spacer_job(db.streams.Repeat, spacing_distance)
    num_rows = 0
    for hist in table.load():
        # Verify outputs are repeated correctly
        if num_rows % spacing_distance == 0:
            ref_hist = hist
        assert len(hist) == 3
        for c in range(len(hist)):
            assert (ref_hist[c] == hist[c]).all()
        num_rows += 1
    assert num_rows == ScannerFrameStream(db, 'test1').len() * spacing_distance

    # Null
    table = run_spacer_job(db.streams.RepeatNull, spacing_distance)
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
    assert num_rows == ScannerFrameStream(db, 'test1').len() * spacing_distance


def test_slice(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    slice_frame = db.streams.Slice(frame, partitions=[db.partitioner.all(50)])
    unsliced_frame = db.streams.Unslice(slice_frame)
    output = ScannerStream(db, 'test_slicing')
    output_op = db.io.Output(unsliced_frame, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    assert input.len() == output.len()


def test_overlapping_slice(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    slice_frame = db.streams.Slice(frame, partitions=[
        db.partitioner.strided_ranges([(0, 15), (5, 25), (15, 35)], 1)])
    sample_frame = db.streams.Range(slice_frame, ranges=[SliceList([
        {'start': 0, 'end': 10},
        {'start': 5, 'end': 15},
        {'start': 5, 'end': 15},
    ])])
    unsliced_frame = db.streams.Unslice(sample_frame)
    output = ScannerStream(db, 'test_slicing')
    output_op = db.io.Output(unsliced_frame, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
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


def test_slice_args(db):
    frame = db.io.Input([ScannerFrameStream(db, 'test1')])
    slice_frame = db.streams.Slice(frame, [db.partitioner.ranges(
        [[0, 1], [1, 2], [2, 3]])])
    test = db.ops.TestSliceArgs(frame=slice_frame, arg=[SliceList([{'arg': i} for i in range(3)])])
    unsliced_frame = db.streams.Unslice(test)
    output = ScannerStream(db, 'test_slicing')
    output_op = db.io.Output(unsliced_frame, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    list(output.load())


def test_bounded_state(db):
    warmup = 3

    frame = db.io.Input([ScannerFrameStream(db, 'test1')])
    increment = db.ops.TestIncrementBounded(ignore=frame, bounded_state=warmup)
    sampled_increment = db.streams.Gather(increment, indices=[[0, 10, 25, 26, 27]])
    output = ScannerStream(db, 'test_bounded_state')
    output_op = db.io.Output(sampled_increment, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    expected_output = [0, warmup, warmup, warmup + 1, warmup + 2]
    for buf in output.load():
        (val, ) = struct.unpack('=q', buf)
        assert val == expected_output[num_rows]
        num_rows += 1
    assert num_rows == 5


def test_unbounded_state(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    slice_frame = db.streams.Slice(frame, partitions=[db.partitioner.all(50)])
    increment = db.ops.TestIncrementUnbounded(ignore=slice_frame)
    unsliced_increment = db.streams.Unslice(increment)
    output = ScannerStream(db, 'test_unbounded_state')
    output_op = db.io.Output(unsliced_increment, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    assert output.len() == input.len()


class DeviceTestBench:
    def test_cpu(self, db):
        self.run(db, DeviceType.CPU)

    @gpu
    def test_gpu(self, db):
        self.run(db, DeviceType.GPU)


class TestHistogram(DeviceTestBench):
    def run(self, db, device):
        input = ScannerFrameStream(db, 'test1')
        frame = db.io.Input([input])
        hist = db.ops.Histogram(frame=frame, device=device)
        output = ScannerStream(db, 'test_hist')
        output_op = db.io.Output(hist, [output])

        db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
        next(output.load())


class TestInplace(DeviceTestBench):
    def run(self, db, device):
        input = ScannerFrameStream(db, 'test1_inplace')
        frame = db.io.Input([input])
        hist = db.ops.Histogram(frame=frame, device=device)
        output = ScannerStream(db, 'test_hist')
        output_op = db.io.Output(hist, [output])

        db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
        next(output.load())


class TestOpticalFlow(DeviceTestBench):
    def run(self, db, device):
        input = ScannerFrameStream(db, 'test1')
        frame = db.io.Input([input])
        flow = db.ops.OpticalFlow(frame=frame, stencil=[-1, 0], device=device)
        flow_range = db.streams.Range(flow, ranges=[{'start': 0, 'end': 50}])
        output = ScannerStream(db, 'test_flow')
        output_op = db.io.Output(flow_range, [output])
        db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
        assert output.len() == 50

        flow_array = next(output.load())
        assert flow_array.dtype == np.float32
        assert flow_array.shape[0] == 480
        assert flow_array.shape[1] == 640
        assert flow_array.shape[2] == 2


def test_stencil(db):
    input = ScannerFrameStream(db, 'test1')

    frame = db.io.Input([input])
    sample_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[-1, 0])
    output = ScannerStream(db, 'test_stencil')
    output_op = db.io.Output(flow, [output])
    db.run(output_op,
           cache_mode=CacheMode.Overwrite,
           show_progress=False,
           pipeline_instances_per_node=1)
    assert output.len() == 1

    frame = db.io.Input([input])
    sample_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output = ScannerStream(db, 'test_stencil')
    output_op = db.io.Output(flow, [output])
    db.run(output_op,
           cache_mode=CacheMode.Overwrite,
           show_progress=False,
           pipeline_instances_per_node=1)

    frame = db.io.Input([input])
    sample_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 2}])
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output = ScannerStream(db, 'test_stencil')
    output_op = db.io.Output(flow, [output])
    db.run(output_op,
           cache_mode=CacheMode.Overwrite,
           show_progress=False,
           pipeline_instances_per_node=1)
    assert output.len() == 2

    frame = db.io.Input([input])
    flow = db.ops.OpticalFlow(frame=frame, stencil=[-1, 0])
    sample_flow = db.streams.Range(flow, ranges=[{'start': 0, 'end': 1}])
    output = ScannerStream(db, 'test_stencil')
    output_op = db.io.Output(sample_flow, [output])
    db.run(output_op,
           cache_mode=CacheMode.Overwrite,
           show_progress=False,
           pipeline_instances_per_node=1)
    assert output.len() == 1


def test_wider_than_packet_stencil(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    sample_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 3}])
    flow = db.ops.OpticalFlow(frame=sample_frame, stencil=[0, 1])
    output = ScannerStream(db, 'test_stencil')
    output_op = db.io.Output(flow, [output])

    db.run(
        output_op,
        cache_mode=CacheMode.Overwrite,
        show_progress=False,
        io_packet_size=1,
        work_packet_size=1,
        pipeline_instances_per_node=1)

    assert output.len() == 3


# def test_packed_file_source(db):
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

#     data = db.sources.PackedFile()
#     pass_data = db.ops.Pass(input=data)
#     output_op = db.sinks.Column(columns={'integer': pass_data})
#     job = Job(op_args={
#         data: {
#             'path': path
#         },
#         output_op: 'test_cpp_source',
#     })

#     db.run(output_op, [job], cache_mode=CacheMode.Overwrite, show_progress=False)
#     tables = [db.table('test_cpp_source')]

#     num_rows = 0
#     for buf in tables[0].column('integer').load():
#         (val, ) = struct.unpack('=Q', buf)
#         assert val == num_rows
#         num_rows += 1
#     assert num_elements == tables[0].num_rows()


def test_files_source(db):
    # Write test files
    path_template = '/tmp/files_source_test_{:d}'
    num_elements = 4
    paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        with open(path, 'wb') as f:
            # Write data
            f.write(struct.pack('=Q', i))
        paths.append(path)

    data = db.io.Input([FilesStream(paths=paths)])
    pass_data = db.ops.Pass(input=data)
    output = ScannerStream(db, 'test_files_source')
    output_op = db.io.Output(pass_data, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    for buf in output.load():
        (val, ) = struct.unpack('=Q', buf)
        assert val == num_rows
        num_rows += 1
    assert num_elements == num_rows


def test_files_sink(db):
    # Write initial test files
    path_template = '/tmp/files_source_test_{:d}'
    num_elements = 4
    input_paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        with open(path, 'wb') as f:
            # Write data
            f.write(struct.pack('=Q', i))
        input_paths.append(path)

    # Write output test files
    path_template = '/tmp/files_sink_test_{:d}'
    num_elements = 4
    output_paths = []
    for i in range(num_elements):
        path = path_template.format(i)
        output_paths.append(path)
    data = db.io.Input([FilesStream(paths=input_paths)])
    pass_data = db.ops.Pass(input=data)
    output = FilesStream(paths=output_paths)
    output_op = db.io.Output(pass_data, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    # Read output test files
    for i, s in enumerate(output.load()):
        d, = struct.unpack('=Q', s)
        assert d == i


def test_python_source(db):
    # Write test files
    py_data = [{'{:d}'.format(i): i} for i in range(4)]

    data = db.io.Input([PythonStream(py_data)])
    pass_data = db.ops.Pass(input=data)
    output = ScannerStream(db, 'test_python_source')
    output_op = db.io.Output(pass_data, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    num_rows = 0
    for i, buf in enumerate(output.load()):
        d = pickle.loads(buf)
        assert d['{:d}'.format(i)] == i
        num_rows += 1
    assert num_rows == 4


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


def test_python_kernel(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 3}])
    test_out = db.ops.TestPy(frame=range_frame, kernel_arg=1, x=[0], y=[0])
    output = ScannerStream(db, 'test_hist')
    output_op = db.io.Output(test_out, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


@scannerpy.register_python_op(batch=50)
class TestPyBatch(Kernel):
    def execute(self, frame: Sequence[FrameType]) -> Sequence[bytes]:
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        input_count = len(frame)
        column_count = 1
        return [point.SerializeToString() for _ in range(input_count)]


def test_python_batch_kernel(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    test_out = db.ops.TestPyBatch(frame=range_frame, batch=50)
    output = ScannerStream(db, 'test_hist')
    output_op = db.io.Output(test_out, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


@scannerpy.register_python_op(stencil=[0, 1])
class TestPyStencil(Kernel):
    def execute(self, frame: Sequence[FrameType]) -> bytes:
        assert len(frame) == 2
        point = protobufs.Point()
        point.x = 10
        point.y = 5
        return point.SerializeToString()


def test_python_stencil_kernel(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    test_out = db.ops.TestPyStencil(frame=range_frame)
    output = ScannerStream(db, 'test_hist')
    output_op = db.io.Output(test_out, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
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


def test_python_stencil_batch_kernel(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    test_out = db.ops.TestPyStencilBatch(frame=range_frame, batch=50)
    output = ScannerStream(db, 'test_hist')
    output_op = db.io.Output(test_out, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_bind_op_args(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input, input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 1} for _ in range(2)])
    test_out = db.ops.TestPy(frame=range_frame, kernel_arg=1, x=[1, 10], y=[5, 50])
    outputs = [ScannerStream(db, 'test_hist_0'), ScannerStream(db, 'test_hist_1')]
    output_op = db.io.Output(test_out, outputs)
    pairs = [(1, 5), (10, 50)]
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    for i, (x, y) in enumerate(pairs):
        values = list(outputs[i].load())
        p = values[0]
        assert p['x'] == x
        assert p['y'] == y


def test_blur(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = ScannerFrameStream(db, 'test_blur')
    output_op = db.io.Output(blurred_frame, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    frame_array = next(output.load())
    assert frame_array.dtype == np.uint8
    assert frame_array.shape[0] == 480
    assert frame_array.shape[1] == 640
    assert frame_array.shape[2] == 3


def test_lossless(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = ScannerFrameStream(db, 'test_blur')
    output_op = db.io.Output(blurred_frame.lossless(), [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_compress(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = ScannerFrameStream(db, 'test_blur')
    output_op = db.io.Output(blurred_frame.compress('video', bitrate=1 * 1024 * 1024), [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)
    next(output.load())


def test_save_mp4(db):
    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 30}])
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output = ScannerFrameStream(db, 'test_save_mp4')
    output_op = db.io.Output(blurred_frame, [output])
    db.run(output_op, cache_mode=CacheMode.Overwrite, show_progress=False)

    f = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    f.close()
    output.save_mp4(f.name)
    run(['rm', '-rf', f.name])


@pytest.fixture()
def no_workers_db():
    # Create new config
    (cfg_path, cfg) = make_config(master_port='5020', worker_port='5021')

    # Setup and ingest video
    with Database(workers=[], config_path=cfg_path, enable_watchdog=False,
                  debug=True) as db:
        (vid1_path, vid2_path) = download_videos()

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield db

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_no_workers(no_workers_db):
    db = no_workers_db

    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    hist = db.ops.Histogram(frame=frame)
    output_op = db.io.Output(hist, [ScannerStream(db, '_ignore')])

    exc = False
    try:
        db.run(output_op, show_progress=False, cache_mode=CacheMode.Overwrite)
    except ScannerException:
        exc = True

    assert exc


@pytest.fixture()
def fault_db():
    # Create new config
    (cfg_path, cfg) = make_config(
        master_port='5010', worker_port='5011', path='/tmp/config_test')

    # Setup and ingest video
    with Database(
            master='localhost:5010',
            workers=[],
            config_path=cfg_path,
            no_workers_timeout=120,
            enable_watchdog=False,
            debug=True) as db:
        (vid1_path, vid2_path) = download_videos()

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield db

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


# def test_clean_worker_shutdown(fault_db):
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

#     master_addr = fault_db._master_address
#     worker_addr = fault_db._worker_addresses[0]
#     shutdown_process = Process(target=worker_shutdown_task,
#                              args=(fault_db.config, master_addr, worker_addr))
#     shutdown_process.daemon = True
#     shutdown_process.start()

#     frame = fault_db.sources.FrameColumn()
#     range_frame = frame.sample()
#     sleep_frame = fault_db.ops.SleepFrame(ignore = range_frame)
#     output_op = fault_db.sinks.Column(columns=[sleep_frame])

#     job = Job(
#         op_args={
#             frame: fault_db.table('test1').column('frame'),
#             range_frame: fault_db.sampler.range(0, 15),
#             output_op: 'test_shutdown',
#         }
#     )
#
#     table = fault_db.run(output_op, [job], pipeline_instances_per_node=1, cache_mode=CacheMode.Overwrite,
#                          show_progress=False)
#     table = table[0]
#     assert len([_ for _, _ in table.column('dummy').load()]) == 15

#     # Shutdown the spawned worker
#     channel = grpc.insecure_channel(
#         'localhost:' + str(spawn_port),
#         options=[('grpc.max_message_length', 24499183 * 2)])
#     worker = fault_db.protobufs.WorkerStub(channel)

#     try:
#         worker.Shutdown(fault_db.protobufs.Empty())
#     except grpc.RpcError as e:
#         status = e.code()
#         if status == grpc.StatusCode.UNAVAILABLE:
#             print('could not shutdown worker!')
#             exit(1)
#         else:
#             raise ScannerException('Worker errored with status: {}'
#                                    .format(status))
#     shutdown_process.join()


def test_fault_tolerance(fault_db):
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

    master_addr = fault_db._master_address
    killer_process = Process(
        target=worker_killer_task, args=(fault_db.config, master_addr))
    killer_process.daemon = True
    killer_process.start()

    input = ScannerFrameStream(db, 'test1')
    frame = fault_db.io.Input([input])
    range_frame = fault_db.streams.Range(frame, ranges=[{'start': 0, 'end': 20}])
    sleep_frame = fault_db.ops.SleepFrame(ignore=range_frame)
    output = ScannerStream(fault_db, 'test_fault')
    output_op = fault_db.io.Output(sleep_frame, [output])

    fault_db.run(
        output_op,
        pipeline_instances_per_node=1,
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
def blacklist_db():
    # Create new config
    (cfg_path, cfg) = make_config(master_port='5055', worker_port='5060')

    # Setup and ingest video
    master = 'localhost:5055'
    workers = ['localhost:{:04d}'.format(5060 + d) for d in range(4)]
    with Database(
            config_path=cfg_path,
            no_workers_timeout=120,
            master=master,
            workers=workers,
            enable_watchdog=False) as db:
        (vid1_path, vid2_path) = download_videos()

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield db

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_job_blacklist(blacklist_db):
    # NOTE(wcrichto): this class must NOT be at the top level. If it is, then pytest injects
    # some of its dependencies, and sending this class to an external Scanner process will fail
    # with a missing "py_test" import..
    @scannerpy.register_python_op()
    class TestPyFail(Kernel):
        def execute(self, frame: FrameType) -> bytes:
            raise ScannerException('Test')

    db = blacklist_db

    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    failed_output = db.ops.TestPyFail(frame=range_frame)
    output = ScannerFrameStream(db, 'test_py_fail')
    output_op = db.io.Output(failed_output, [output])
    db.run(
        output_op,
        cache_mode=CacheMode.Overwrite,
        show_progress=False,
        pipeline_instances_per_node=1)
    assert not output.committed()


@pytest.fixture()
def timeout_db():
    # Create new config
    (cfg_path, cfg) = make_config(master_port='5155', worker_port='5160')

    # Setup and ingest video
    master = 'localhost:5155'
    workers = ['localhost:{:04d}'.format(5160 + d) for d in range(4)]
    with Database(
            config_path=cfg_path,
            no_workers_timeout=120,
            master=master,
            workers=workers,
            enable_watchdog=False) as db:
        (vid1_path, vid2_path) = download_videos()

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield db

        for worker in workers:
            channel = grpc.insecure_channel(worker)
            worker_stub = protobufs.WorkerStub(channel)
            try:
                worker_stub.Shutdown(
                    protobufs.Empty(), timeout=db._grpc_timeout)
            except grpc.RpcError as e:
                pass

        # Tear down
        run([
            'rm', '-rf', cfg['storage']['db_path'], cfg_path, vid1_path,
            vid2_path
        ])


def test_job_timeout(timeout_db):
    @scannerpy.register_python_op()
    def timeout_fn(self, frame: FrameType) -> bytes:
        time.sleep(5)
        return bytes('what', 'utf-8')

    db = timeout_db

    input = ScannerFrameStream(db, 'test1')
    frame = db.io.Input([input])
    range_frame = db.streams.Range(frame, ranges=[{'start': 0, 'end': 1}])
    sleep_frame = db.ops.timeout_fn(frame=range_frame)
    output = ScannerFrameStream(db, 'test_timeout')
    output_op = db.io.Output(sleep_frame, [output])

    db.run(
        output_op,
        pipeline_instances_per_node=1,
        task_timeout=0.1,
        cache_mode=CacheMode.Overwrite,
        show_progress=False)

    assert not output.committed()


@pytest.fixture(scope='module')
def sql_db(db):
    with Postgresql() as postgresql:
        conn = psycopg2.connect(**postgresql.dsn())
        cur = conn.cursor()

        cur.execute(
            'CREATE TABLE test (id serial PRIMARY KEY, a integer, b integer, c text, d varchar(255), e boolean, f float, grp integer)'
        )
        cur.execute(
            "INSERT INTO test (a, b, c, d, e, f, grp) VALUES (10, 0, 'hi', 'hello', true, 2.0, 0)"
        )
        cur.execute(
            "INSERT INTO test (a, b, c, d, e, f, grp) VALUES (20, 0, 'hi', 'hello', true, 2.0, 0)"
        )
        cur.execute(
            "INSERT INTO test (a, b, c, d, e, f, grp) VALUES (30, 0, 'hi', 'hello', true, 2.0, 1)"
        )
        cur.execute('CREATE TABLE jobs (id serial PRIMARY KEY, name text)')
        cur.execute(
            'CREATE TABLE test2 (id serial PRIMARY KEY, b integer, s text)')
        conn.commit()

        sql_params = postgresql.dsn()
        sql_config = protobufs.SQLConfig(
            hostaddr=sql_params['host'],
            port=sql_params['port'],
            dbname=sql_params['database'],
            user=sql_params['user'],
            adapter='postgres')

        yield db, SQLStorage(config=sql_config, job_table='jobs'), cur

        cur.close()
        conn.close()


@scannerpy.register_python_op(name='AddOne')
def add_one(config, row: bytes) -> bytes:
    row = json.loads(row.decode('utf-8'))
    return json.dumps([{'id': r['id'], 'b': r['a'] + 1} for r in row])


def test_sql(sql_db):
    (db, storage, cur) = sql_db

    cur.execute('SELECT COUNT(*) FROM test');
    n, = cur.fetchone()

    row = db.io.Input([SQLInputStream(
        query=protobufs.SQLQuery(
            fields='test.id as id, test.a, test.c, test.d, test.e, test.f',
            table='test',
            id='test.id',
            group='test.id'),
        filter='true',
        storage=storage,
        num_elements=n)])
    row2 = db.ops.AddOne(row=row)
    output_op = db.io.Output(row2, [SQLOutputStream(
        table='test',
        storage=storage,
        job_name='foobar',
        insert=False)])
    db.run(output_op)

    cur.execute('SELECT b FROM test')
    assert cur.fetchone()[0] == 11

    cur.execute('SELECT name FROM jobs')
    assert cur.fetchone()[0] == 'foobar'

@scannerpy.register_python_op(name='AddAll')
def add_all(config, row: bytes) -> bytes:
    row = json.loads(row.decode('utf-8'))
    total = sum([r['a'] for r in row])
    return json.dumps([{'id': r['id'], 'b': total} for r in row])


def test_sql_grouped(sql_db):
    (db, storage, cur) = sql_db

    row = db.io.Input([SQLInputStream(
        storage=storage,
        query=protobufs.SQLQuery(
            fields='test.id as id, test.a',
            table='test',
            id='test.id',
            group='test.grp'),
        filter='true')])
    row2 = db.ops.AddAll(row=row)
    output_op = db.io.Output(
        row2, [SQLOutputStream(storage=storage, table='test', job_name='test', insert=False)])
    db.run(output_op)

    cur.execute('SELECT b FROM test')
    assert cur.fetchone()[0] == 30


@scannerpy.register_python_op(name='SQLInsertTest')
def sql_insert_test(config, row: bytes) -> bytes:
    row = json.loads(row.decode('utf-8'))
    return json.dumps([{'s': 'hello world', 'b': r['a'] + 1} for r in row])


def test_sql_insert(sql_db):
    (db, storage, cur) = sql_db

    row = db.io.Input([SQLInputStream(
        storage=storage,
        query=protobufs.SQLQuery(
            fields='test.id as id, test.a',
            table='test',
            id='test.id',
            group='test.grp'),
        filter='true')])
    row2 = db.ops.SQLInsertTest(row=row)
    output_op = db.io.Output(
        row2, [SQLOutputStream(
            storage=storage, table='test2', job_name='test', insert=True)])
    db.run(output_op, show_progress=False)

    cur.execute('SELECT s FROM test2')
    assert cur.fetchone()[0] == "hello world"


def test_audio(db):
    (vid_path, _) = download_videos()
    audio = db.io.Input([AudioStream(vid_path, 1.0)])
    ignored = db.ops.DiscardFrame(ignore=audio)
    output = db.io.Output(ignored, [ScannerStream(db, 'audio_test')])
    db.run(output, cache_mode=CacheMode.Overwrite)


def download_transcript():
    url = "https://storage.googleapis.com/scanner-data/test/transcript.cc1.srt"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.cc1.srt') as f:
        resp = requests.get(url, stream=True)
        assert resp.ok
        for block in resp.iter_content(1024):
            f.write(block)
        return f.name

@scannerpy.register_python_op(name='DecodeCap')
def decode_cap(config, cap: bytes) -> bytes:
    cap = json.loads(cap.decode('utf-8'))
    return b' '


def test_captions(db):
    caption_path = download_transcript()
    captions = db.io.Input([CaptionStream(caption_path, window_size=10.0, max_time=3600)])
    ignored = db.ops.DecodeCap(cap=captions)
    output = db.io.Output(ignored, [ScannerStream(db, 'caption_test')])
    db.run(output, cache_mode=CacheMode.Overwrite, pipeline_instances_per_node=1)


@scannerpy.register_python_op(name='CacheTest')
def cache_test(config, n: Any) -> Any:
    return n + 1


def test_cache_mode(db):
    n = db.io.Input([PythonStream([0])])
    out = db.ops.CacheTest(n=n)
    output = ScannerStream(db, 'test_cache')
    output_op = db.io.Output(out, [output])
    db.run(output_op)

    assert next(output.load()) == 1

    exc = False
    try:
        db.run(output_op)
    except ScannerException:
        exc = True
    assert exc

    n = db.io.Input([PythonStream([1])])
    out = db.ops.CacheTest(n=n)
    output_op = db.io.Output(out, [output])

    db.run(output_op, cache_mode=CacheMode.Ignore)
    assert next(output.load()) == 1

    db.run(output_op, cache_mode=CacheMode.Overwrite)
    assert next(output.load()) == 2


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
