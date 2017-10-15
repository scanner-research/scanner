from scannerpy import (
    Database, Config, DeviceType, ColumnType, BulkJob, Job, ProtobufGenerator)
from scannerpy.stdlib import parsers
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

try:
    run(['nvidia-smi'])
    has_gpu = True
except OSError:
    has_gpu = False

gpu = pytest.mark.skipif(
    not has_gpu,
    reason='need GPU to run')
slow = pytest.mark.skipif(
    not pytest.config.getoption('--runslow'),
    reason='need --runslow option to run')

cwd = os.path.dirname(os.path.abspath(__file__))

@slow
def test_tutorial():
    def run_py(path):
        print(path)
        run(
            'cd {}/../examples/tutorial && python {}.py'.format(cwd, path),
            shell=True)

    run(
        'cd {}/../examples/tutorial/resize_op && '
        'mkdir -p build && cd build && cmake -D SCANNER_PATH={} .. && '
        'make'.format(cwd, cwd + '/..'),
        shell=True)

    tutorials = [
        '00_basic',
        '01_sampling',
        '02_collections',
        '03_ops',
        '04_compression',
        '05_custom_op']

    for t in tutorials:
        run_py(t)

@slow
def test_examples():
    def run_py((d, f)):
        print(f)
        run('cd {}/../examples/{} && python {}.py'.format(cwd, d, f),
            shell=True)

    examples = [
        ('face_detection', 'face_detect'),
        ('shot_detection', 'shot_detect')]

    for e in examples:
        run_py(e)

@pytest.fixture(scope="module")
def db():
    # Create new config
    with tempfile.NamedTemporaryFile(delete=False) as f:
        cfg = Config.default_config()
        cfg['storage']['db_path'] = tempfile.mkdtemp()
        f.write(toml.dumps(cfg))
        cfg_path = f.name

    # Setup and ingest video
    with Database(config_path=cfg_path, debug=True) as db:
        # Download video from GCS
        url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            host = socket.gethostname()
            # HACK: special proxy case for Ocean cluster
            if host in ['ocean', 'crissy', 'pismo', 'stinson']:
                resp = requests.get(url, stream=True, proxies={
                    'https': 'http://proxy.pdl.cmu.edu:3128/'
                })
            else:
                resp = requests.get(url, stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            vid1_path = f.name

        # Make a second one shorter than the first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            vid2_path = f.name
        run(['ffmpeg', '-y', '-i', vid1_path, '-ss', '00:00:00', '-t',
             '00:00:10', '-c:v', 'libx264', '-strict', '-2', vid2_path])

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield db

        # Tear down
        run(['rm', '-rf',
            cfg['storage']['db_path'],
            cfg_path,
            vid1_path,
            vid2_path])

def test_new_database(db): pass

def test_table_properties(db):
    table = db.table('test1')
    assert table.id() == 0
    assert table.name() == 'test1'
    assert table.num_rows() == 720
    assert [c.name() for c in table.columns()] == ['index', 'frame']

# def test_collection(db):
#     c = db.new_collection('test', ['test1', 'test2'])
#     frame = c.as_op().strided(2)
#     job = Job(
#         columns = [db.ops.Histogram(frame = frame)],
#         name = '_ignore')
#     db.run(job, show_progress=False, force=True)
#     db.delete_collection('test')

def test_summarize(db):
    db.summarize()

def test_load_video_column(db):
    next(db.table('test1').load(['frame']))
    # Gather rows
    rows = [0, 10, 100, 200]
    frames = [_ for _ in db.table('test1').load(['frame'], rows=rows)]
    assert len(frames) == len(rows)

def test_load_video_column(db):
    next(db.table('test1').load(['frame']))

def test_profiler(db):
    frame = db.ops.FrameInput()
    hist = db.ops.Histogram(frame=frame)
    output_op = db.ops.Output(columns=[hist])

    job = Job(
        output_table_name='_ignore',
        op_args={
            frame: db.table('test1').column('frame')
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])

    output = db.run(bulk_job, show_progress=False, force=True)
    profiler = output[0].profiler()
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    profiler.write_trace(f.name)
    profiler.statistics()
    run(['rm', '-f', f.name])

def builder(cls):
    inst = cls()

    class Generated:
        def test_cpu(self, db):
            inst.run(db, inst.job(db, DeviceType.CPU))

        @gpu
        def test_gpu(self, db):
            inst.run(db, inst.job(db, DeviceType.GPU))

    return Generated

@builder
class TestHistogram:
    def job(self, db, ty):
        frame = db.ops.FrameInput()
        hist = db.ops.Histogram(frame=frame, device=ty)
        output_op = db.ops.Output(columns=[hist])

        job = Job(
            output_table_name='test_hist',
            op_args={
                frame: db.table('test1').column('frame')
            }
        )
        bulk_job = BulkJob(dag=output_op, jobs=[job])
        return bulk_job

    def run(self, db, job):
        tables = db.run(job, force=True, show_progress=False)
        next(tables[0].load(['histogram'], parsers.histograms))

# @builder
# class TestOpticalFlow:
#     def job(self, db, ty):
#         frame = db.table('test1').as_op().range(0, 50, warmup_size=1)
#         flow = db.ops.OpticalFlow(
#             frame = frame,
#             device = ty)
#         return Job(columns = [flow], name = 'test_flow')

#     def run(self, db, job):
#         table = db.run(job, force=True, show_progress=False)
#         fid, flows = next(table.load(['flow']))
#         flow_array = flows[0]
#         assert fid == 0
#         assert flow_array.dtype == np.float32
#         assert flow_array.shape[0] == 480
#         assert flow_array.shape[1] == 640
#         assert flow_array.shape[2] == 2

def test_python_kernel(db):
    db.register_op('TestPy',
                   [('frame', ColumnType.Video)],
                   ['dummy'])
    db.register_python_kernel('TestPy', DeviceType.CPU,
                              cwd + '/test_py_kernel.py')

    frame = db.ops.FrameInput()
    range_frame = frame.sample()
    test_out = db.ops.TestPy(frame=range_frame)
    output_op = db.ops.Output(columns=[test_out])

    job = Job(
        output_table_name='test_hist',
        op_args={
            frame: db.table('test1').column('frame'),
            range_frame: db.sampler.range(0, 30),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])

    tables = db.run(bulk_job, force=True, show_progress=False)
    next(tables[0].load(['dummy']))

def test_blur(db):
    frame = db.ops.FrameInput()
    range_frame = frame.sample()
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output_op = db.ops.Output(columns=[blurred_frame])

    job = Job(
        output_table_name='test_blur',
        op_args={
            frame: db.table('test1').column('frame'),
            range_frame: db.sampler.range(0, 30),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    tables = db.run(bulk_job, force=True, show_progress=False)
    table = tables[0]

    fid, frames = next(table.load(['frame']))
    frame_array = frames[0]
    assert fid == 0
    assert frame_array.dtype == np.uint8
    assert frame_array.shape[0] == 480
    assert frame_array.shape[1] == 640
    assert frame_array.shape[2] == 3

def test_lossless(db):
    frame = db.ops.FrameInput()
    range_frame = frame.sample()
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output_op = db.ops.Output(columns=[blurred_frame.lossless()])

    job = Job(
        output_table_name='test_blur_lossless',
        op_args={
            frame: db.table('test1').column('frame'),
            range_frame: db.sampler.range(0, 30),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    tables = db.run(bulk_job, force=True, show_progress=False)
    table = tables[0]
    next(table.load(['frame']))

def test_compress(db):
    frame = db.ops.FrameInput()
    range_frame = frame.sample()
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    compressed_frame = blurred_frame.compress(
        'video', bitrate = 1 * 1024 * 1024)
    output_op = db.ops.Output(columns=[compressed_frame])

    job = Job(
        output_table_name='test_blur_compressed',
        op_args={
            frame: db.table('test1').column('frame'),
            range_frame: db.sampler.range(0, 30),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    tables = db.run(bulk_job, force=True, show_progress=False)
    table = tables[0]
    next(table.load(['frame']))

def test_save_mp4(db):
    frame = db.ops.FrameInput()
    range_frame = frame.sample()
    blurred_frame = db.ops.Blur(frame=range_frame, kernel_size=3, sigma=0.1)
    output_op = db.ops.Output(columns=[blurred_frame])

    job = Job(
        output_table_name='test_save_mp4',
        op_args={
            frame: db.table('test1').column('frame'),
            range_frame: db.sampler.range(0, 30),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    tables = db.run(bulk_job, force=True, show_progress=False)
    table = tables[0]
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    f.close()
    table.columns('frame').save_mp4(f.name)
    run(['rm', '-rf', f.name])

@pytest.fixture()
def fault_db():
    # Create new config
    #with tempfile.NamedTemporaryFile(delete=False) as f:
    with open('/tmp/config_test', 'w') as f:
        cfg = Config.default_config()
        cfg['storage']['db_path'] = tempfile.mkdtemp()
        cfg['network']['master_port'] = '5005'
        cfg['network']['worker_port'] = '5006'
        f.write(toml.dumps(cfg))
        cfg_path = f.name

    # Setup and ingest video
    with Database(config_path=cfg_path, debug=True) as db:
        # Download video from GCS
        url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            host = socket.gethostname()
            # HACK: special proxy case for Ocean cluster
            if host in ['ocean', 'crissy', 'pismo', 'stinson']:
                resp = requests.get(url, stream=True, proxies={
                    'https': 'http://proxy.pdl.cmu.edu:3128/'
                })
            else:
                resp = requests.get(url, stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            vid1_path = f.name

        # Make a second one shorter than the first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            vid2_path = f.name
        run(['ffmpeg', '-y', '-i', vid1_path, '-ss', '00:00:00', '-t',
             '00:00:10', '-c:v', 'libx264', '-strict', '-2', vid2_path])

        db.ingest_videos([('test1', vid1_path), ('test2', vid2_path)])

        yield db

        # Tear down
        run(['rm', '-rf',
            cfg['storage']['db_path'],
            cfg_path,
            vid1_path,
            vid2_path])


def test_clean_worker_shutdown(fault_db):
    spawn_port = 5010
    def worker_shutdown_task(config, master_address, worker_address):
        from scannerpy import ProtobufGenerator, Config, start_worker
        import time
        import grpc
        import subprocess

        c = Config(None)

        import scanner.metadata_pb2 as metadata_types
        import scanner.engine.rpc_pb2 as rpc_types
        import scanner.types_pb2 as misc_types
        import libscanner as bindings

        protobufs = ProtobufGenerator(config)

        # Wait to kill worker
        time.sleep(8)
        # Kill worker
        channel = grpc.insecure_channel(
            worker_address,
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
                raise ScannerException('Worker errored with status: {}'
                                       .format(status))

        # Wait a bit
        time.sleep(15)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        subprocess.call(['python ' +  script_dir + '/spawn_worker.py'],
                        shell=True)

    master_addr = fault_db._master_address
    worker_addr = fault_db._worker_addresses[0]
    shutdown_process = Process(target=worker_shutdown_task,
                             args=(fault_db.config, master_addr, worker_addr))
    shutdown_process.daemon = True
    shutdown_process.start()

    frame = fault_db.ops.FrameInput()
    range_frame = frame.sample()
    sleep_frame = fault_db.ops.SleepFrame(ignore = range_frame)
    output_op = fault_db.ops.Output(columns=[sleep_frame])

    job = Job(
        output_table_name='test_shutdown',
        op_args={
            frame: fault_db.table('test1').column('frame'),
            range_frame: fault_db.sampler.range(0, 15),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    table = fault_db.run(bulk_job, pipeline_instances_per_node=1, force=True,
                         show_progress=False)
    table = table[0]
    assert len([_ for _, _ in table.column('dummy').load()]) == 15

    # Shutdown the spawned worker
    channel = grpc.insecure_channel(
        'localhost:' + str(spawn_port),
        options=[('grpc.max_message_length', 24499183 * 2)])
    worker = fault_db.protobufs.WorkerStub(channel)

    try:
        worker.Shutdown(fault_db.protobufs.Empty())
    except grpc.RpcError as e:
        status = e.code()
        if status == grpc.StatusCode.UNAVAILABLE:
            print('could not shutdown worker!')
            exit(1)
        else:
            raise ScannerException('Worker errored with status: {}'
                                   .format(status))
    shutdown_process.join()


def test_fault_tolerance(fault_db):
    spawn_port = 5010
    def worker_killer_task(config, master_address, worker_address):
        from scannerpy import ProtobufGenerator, Config, start_worker
        import time
        import grpc
        import subprocess
        import signal
        import os

        c = Config(None)

        import scanner.metadata_pb2 as metadata_types
        import scanner.engine.rpc_pb2 as rpc_types
        import scanner.types_pb2 as misc_types
        import libscanner as bindings

        protobufs = ProtobufGenerator(config)

        # Kill worker
        channel = grpc.insecure_channel(
            worker_address,
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
                raise ScannerException('Worker errored with status: {}'
                                       .format(status))

        # Spawn a worker that we will force kill
        script_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.devnull, 'w') as fp:
            p = subprocess.Popen(
                ['python ' +  script_dir + '/spawn_worker.py'],
                shell=True,
                stdout=fp, stderr=fp,
                preexec_fn=os.setsid)

            # Wait a bit for the worker to do its thing
            time.sleep(10)

            # Force kill worker process to trigger fault tolerance
            os.killpg(os.getpgid(p.pid), signal.SIGTERM) 
            p.communicate()

            # Wait for fault tolerance to kick in
            time.sleep(25)

            # Spawn the worker again
            subprocess.call(['python ' +  script_dir + '/spawn_worker.py'],
                            shell=True)

    master_addr = fault_db._master_address
    worker_addr = fault_db._worker_addresses[0]
    killer_process = Process(target=worker_killer_task,
                             args=(fault_db.config, master_addr, worker_addr))
    killer_process.daemon = True
    killer_process.start()

    frame = fault_db.ops.FrameInput()
    range_frame = frame.sample()
    sleep_frame = fault_db.ops.SleepFrame(ignore = range_frame)
    output_op = fault_db.ops.Output(columns=[sleep_frame])

    job = Job(
        output_table_name='test_fault',
        op_args={
            frame: fault_db.table('test1').column('frame'),
            range_frame: fault_db.sampler.range(0, 20),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    table = fault_db.run(bulk_job, pipeline_instances_per_node=1, force=True,
                         show_progress=False)
    table = table[0]

    assert len([_ for _, _ in table.column('dummy').load()]) == 20

    # Shutdown the spawned worker
    channel = grpc.insecure_channel(
        'localhost:' + str(spawn_port),
        options=[('grpc.max_message_length', 24499183 * 2)])
    worker = fault_db.protobufs.WorkerStub(channel)

    try:
        worker.Shutdown(fault_db.protobufs.Empty())
    except grpc.RpcError as e:
        status = e.code()
        if status == grpc.StatusCode.UNAVAILABLE:
            print('could not shutdown worker!')
            exit(1)
        else:
            raise ScannerException('Worker errored with status: {}'
                                   .format(status))
    killer_process.join()
