from scannerpy import Database, Config, DeviceType, Job
from scannerpy.stdlib import parsers
import tempfile
import toml
import pytest
from subprocess import check_call as run
import requests
import imp
import os.path
import socket
import numpy as np

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
    with Database(config_path=cfg_path) as db:
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

def test_collection(db):
    c = db.new_collection('test', ['test1', 'test2'])
    frame = c.as_op().strided(2)
    job = Job(
        columns = [db.ops.Histogram(frame = frame)],
        name = '_ignore')
    db.run(job, show_progress=False, force=True)
    db.delete_collection('test')

def test_summarize(db):
    db.summarize()

def test_load_video_column(db):
    next(db.table('test1').load(['frame']))

def test_profiler(db):
    frame = db.table('test1').as_op().all()
    job = Job(
        columns = [db.ops.Histogram(frame = frame)],
        name = '_ignore')
    output = db.run(job, show_progress=False, force=True)
    profiler = output.profiler()
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
        frame = db.table('test1').as_op().all()
        histogram = db.ops.Histogram(frame = frame, device = ty)
        return Job(columns = [histogram], name = 'test_hist')

    def run(self, db, job):
        table = db.run(job, force=True, show_progress=False)
        next(table.load([1], parsers.histograms))

@builder
class TestOpticalFlow:
    def job(self, db, ty):
        frame = db.table('test1').as_op().range(0, 50, warmup_size=1)
        flow = db.ops.OpticalFlow(
            frame = frame,
            device = ty)
        return Job(columns = [flow], name = 'test_flow')

    def run(self, db, job):
        table = db.run(job, force=True, show_progress=False)
        fid, flows = next(table.load(['flow']))
        flow_array = flows[0]
        assert fid == 0
        assert flow_array.dtype == np.float32
        assert flow_array.shape[0] == 480
        assert flow_array.shape[1] == 640
        assert flow_array.shape[2] == 2

def test_blur(db):
    frame = db.table('test1').as_op().range(0, 30)
    blurred_frame = db.ops.Blur(frame = frame, kernel_size = 3)
    job = Job(columns = [blurred_frame], name = 'test_blur')
    table = db.run(job, force=True, show_progress=False)
    fid, frames = next(table.load(['frame']))
    frame_array = frames[0]
    assert fid == 0
    assert frame_array.dtype == np.uint8
    assert frame_array.shape[0] == 480
    assert frame_array.shape[1] == 640
    assert frame_array.shape[2] == 3

def test_lossless(db):
    frame = db.table('test1').as_op().range(0, 30)
    blurred_frame = db.ops.Blur(frame = frame, kernel_size = 3, sigma = 0.1)
    job = Job(columns = [blurred_frame.lossless()],
              name = 'test_blur_lossless')
    table = db.run(job, force=True, show_progress=False)
    next(table.load(['frame']))

def test_compress(db):
    frame = db.table('test1').as_op().range(0, 30)
    blurred_frame = db.ops.Blur(frame = frame, kernel_size = 3, sigma = 0.1)
    compressed_frame = blurred_frame.compress(
        'video', bitrate = 1 * 1024 * 1024)
    job = Job(columns = [compressed_frame], name = 'test_blur_compressed')
    table = db.run(job, force=True, show_progress=False)
    next(table.load(['frame']))

def test_save_mp4(db):
    frame = db.table('test1').as_op().range(0, 30, item_size=10)
    blurred_frame = db.ops.Blur(frame = frame, kernel_size = 3, sigma = 0.1)
    job = Job(columns = [blurred_frame], name = 'test_save_mp4')
    table = db.run(job, force=True, show_progress=False)
    f = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    f.close()
    table.columns('frame').save_mp4(f.name)
    run(['rm', '-rf', f.name])
