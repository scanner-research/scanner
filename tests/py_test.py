from scannerpy import Database, Config, DeviceType
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
    db = Database(config_path=cfg_path, debug=True)
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
    assert len(table.columns()) == 2
    assert [c.name() for c in table.columns()] == ['frame', 'frame_info']

def test_make_collection(db):
    db.new_collection('test', ['test'])

def test_load_video_column(db):
    next(db.table('test').columns(0).load())

def test_profiler(db):
    [output] = db.run(
        db.sampler().all([('test', '_ignore')]),
        db.ops.Histogram(),
        show_progress=False)
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
            inst.run(db, inst.op(db, DeviceType.CPU))

        @gpu
        def test_gpu(self, db):
            inst.run(db, inst.op(db, DeviceType.GPU))

    return Generated

@builder
class TestHistogram:
    def op(self, db, ty):
        return db.ops.Histogram(device=ty)

    def run(self, db, op):
        [table] = db.run(db.sampler().all([('test', 'test_hist')]), op,
                         force=True, show_progress=False)
        next(table.load([0], parsers.histograms))

@builder
class TestOpticalFlow:
    def op(self, db, ty):
        input = db.ops.Input()
        flow = db.ops.OpticalFlow(
            inputs=[(input,['frame', 'frame_info'])],
            device=ty)
        output = db.ops.Output(inputs=[(flow, ['flow']), (input, ['frame_info'])])
        return output

    def run(self, db, op):
        tasks = db.sampler().range([('test', 'test_flow')], 0, 50, warmup_size=1)
        [table] = db.run(tasks, op, force=True, show_progress=False)
        next(table.load([0, 1], parsers.flow))
