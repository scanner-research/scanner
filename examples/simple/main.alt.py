from scannerpy import Database, Evaluator, DeviceType, NetDescriptor
import numpy as np
import cv2

db = Database()

# hist = db.evaluators.Histogram(device = DeviceType.GPU)
descriptor = NetDescriptor.from_file(db, 'features/googlenet.toml')
caffe_args = {
    'device': DeviceType.GPU,
    'net_descriptor': descriptor.as_proto(),
    'batch_size': 96
}
table_input = db.evaluators.Input()
caffe_input = db.evaluators.CaffeInput(
    inputs=[(table_input, ["frame", "frame_info"])],
    **caffe_args)
caffe = db.evaluators.Caffe(
    inputs=[(caffe_input, ["caffe_frame"]), (table_input, ["frame_info"])],
    **caffe_args)

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
        'meangirls', ['/n/scanner/wcrichto.new/videos/meanGirls_short.mp4'],
        force=True)
    input_collection = db.collection('meangirls')
    sampler = db.sampler()
    strided = sampler.strided(input_collection, 1)
    output_collection = db.run(strided, caffe, 'meangirls_hist')
    table = output_collection.tables(0)
    print [x for (x, _) in table.columns(0).load(parse_hist)]
    db.profiler(0).write_trace('test.trace')


video_collection()
