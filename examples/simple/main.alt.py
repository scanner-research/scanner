from scannerpy import Database, DeviceType
import numpy as np

db = Database()

hist = db.ops.Histogram(device=DeviceType.GPU)

def parse_hist(buf):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)

def single_video():
    video = '/bigdata/wcrichto/videos/meanGirls_short.mp4'
    db.ingest_video(('meangirls', video))
    sampler = db.sampler()
    tasks = sampler.all([('meangirls', 'meangirls_hist')])
    [table] = db.run(tasks, hist)

def video_collection():
    input_collection, _ = db.ingest_video_collection(
        'meangirls',
        ['/bigdata/wcrichto/videos/meanGirls_short.mp4'],
        force=True)
    input_collection = db.collection('meangirls')
    sampler = db.sampler()
    tasks = sampler.range(input_collection, 0, 100)
    output_collection = db.run(tasks, hist, 'meangirls_hist', force=True)
    table = output_collection.tables(0)
    output_collection.profiler().write_trace('test.trace')



video_collection()
