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
    input_collection = db.ingest_video_collection(
        'meangirls',
        ['/n/scanner/wcrichto.new/videos/meanGirls_short.mp4'],
        force=True)
    input_collection = db.collection('meangirls')
    sampler = db.sampler()
    strided = sampler.strided(input_collection, 4)
    output_collection = db.run(strided, hist, 'meangirls_hist', force=True)
    table = output_collection.tables(0)
    output_collection.profiler().write_trace('test.trace')



video_collection()
