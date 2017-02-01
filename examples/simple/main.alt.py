from scannerpy import Database, Evaluator, DeviceType
import numpy as np
import cv2

db = Database()

hist = db.evaluators.Histogram(device = DeviceType.CPU)

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
        'meangirls', ['/bigdata/wcrichto/videos/meanGirls_short.mp4'])
    sampler = db.sampler()
    strided = sampler.strided(input_collection, 2)
    output_collection = db.run(strided, hist, 'meangirls_hist')
    table = output_collection.tables(0)
    print [x for (x, _) in table.columns(0).load(parse_hist)]

video_collection()
