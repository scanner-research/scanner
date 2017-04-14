from scannerpy import Database, Config, DeviceType, Job
from scannerpy.stdlib import parsers
import tempfile
import toml
import pytest
import subprocess
import requests
import imp
import os.path

try:
    subprocess.check_call(['nvidia-smi'])
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

# def test_tutorial():
#     def run_py(path):
#         print(path)
#         subprocess.check_call(
#             'cd {}/../examples/tutorial && python {}.py'.format(cwd, path),
#             shell=True)

#     subprocess.check_call(
#         'cd {}/../examples/tutorial/resize_op && '
#         'mkdir -p build && cd build && cmake -D SCANNER_PATH={} .. && '
#         'make'.format(cwd, cwd + '/..'),
#         shell=True)

#     tutorials = [
#         '00_basic',
#         '01_sampling',
#         '02_collections',
#         '03_ops',
#         '04_custom_op']

#     for t in tutorials:
#         run_py(t)

@slow
def test_examples():
    def run_py((d, f)):
        print(f)
        subprocess.check_call(
            'cd {}/../examples/{} && python {}.py'.format(cwd, d, f),
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
        url = "https://storage.googleapis.com/scanner-data/test/short_video.mp4"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            resp = requests.get(url, stream=True)
            assert resp.ok
            for block in resp.iter_content(1024):
                f.write(block)
            vid_path = f.name
        db.ingest_videos([('test', vid_path)])

        yield db

        # Tear down
        subprocess.check_call(['rm', '-rf',
                               cfg['storage']['db_path'],
                               cfg_path,
                               vid_path])

def test_new_database(db): pass

def test_table_properties(db):
    table = db.table('test')
    assert table.id() == 0
    assert table.name() == 'test'
    assert table.num_rows() == 720
    assert [c.name() for c in table.columns()] == ['index', 'frame', 'frame_info']

def test_make_collection(db):
    db.new_collection('test', ['test'])

def test_load_video_column(db):
    next(db.table('test').load(['frame']))

def test_profiler(db):
    frame, frame_info = db.table('test').as_op().all()
    job = Job(
        columns = [db.ops.Histogram(frame = frame, frame_info = frame_info)],
        name = '_ignore')
    output = db.run(job, show_progress=False)
    profiler = output.profiler()
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()
    profiler.write_trace(f.name)
    profiler.statistics()
    subprocess.check_call(['rm', '-f', f.name])

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
        frame, frame_info = db.table('test').as_op().all()
        histogram = db.ops.Histogram(frame = frame, frame_info = frame_info, device = ty)
        return Job(columns = [histogram], name = 'test_hist')

    def run(self, db, job):
        table = db.run(job, force=True, show_progress=False)
        next(table.load([1], parsers.histograms))

@builder
class TestOpticalFlow:
    def job(self, db, ty):
        frame, frame_info = db.table('test').as_op().range(0, 50, warmup_size=1)
        flow = db.ops.OpticalFlow(
            frame = frame, frame_info = frame_info,
            device = ty)
        return Job(columns = [flow, frame_info], name = 'test_flow')

    def run(self, db, job):
        table = db.run(job, force=True, show_progress=False)
        next(table.load(['flow', 'frame_info'], parsers.flow))
