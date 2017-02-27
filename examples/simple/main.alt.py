from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
from sklearn.preprocessing import normalize
import numpy as np
import cv2

db = Database()

hist = db.ops.Histogram(device=DeviceType.CPU)

input = db.ops.Input()
flow = db.ops.OpticalFlow(
    inputs=[(input,['frame', 'frame_info'])],
    device=DeviceType.GPU)
output = db.ops.Output(inputs=[(flow, ['flow']), (input, ['frame_info'])])

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
    tasks = sampler.all(input_collection, warmup_size=1)
    output_collection = db.run(tasks, output, 'meangirls_hist', force=True)
    output_collection = db.collection('meangirls_hist')
    table = output_collection.tables(0)

    vid = cv2.VideoWriter(
        'test.mkv',
        cv2.VideoWriter_fourcc(*'X264'),
        24.0,
        (640, 480))

    for row, flow in table.load((0, 1), parsers.flow):
        img = np.linalg.norm(flow, axis=(2,))*4
        normalize(img)
        img = img.astype(np.uint8)
        vid.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))

    # output_collection.profiler().write_trace('test.trace')



video_collection()
