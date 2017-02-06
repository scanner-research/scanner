from scannerpy import Database, DeviceType, NetDescriptor
from functools import partial
import numpy as np
import cv2

db = Database()

descriptor = NetDescriptor.from_file(db, 'features/caffe_facenet.toml')
facenet_args = db.protobufs.FacenetArgs()
facenet_args.scale = 1.0
facenet_args.threshold = 0.5
caffe_args = facenet_args.caffe_args
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 96

table_input = db.evaluators.Input()
caffe_input = db.evaluators.FacenetInput(
    inputs=[(table_input, ["frame", "frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
caffe = db.evaluators.Facenet(
    inputs=[(caffe_input, ["caffe_frame"]), (table_input, ["frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
caffe_output = db.evaluators.FacenetOutput(
    inputs=[(caffe, ["caffe_output"]), (table_input, ["frame_info"])],
    args=facenet_args)

hist = db.evaluators.Histogram(device=DeviceType.GPU)

def parse_hist(buf):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)

def parse_bboxes(db, buf):
    import struct
    (num_bboxes,) = struct.unpack("=Q", buf[:8])
    buf = buf[8:]
    bboxes = []
    for i in range(num_bboxes):
        (bbox_size,) = struct.unpack("=i", buf[:4])
        buf = buf[4:]
        box = db.protobufs.BoundingBox()
        box.ParseFromString(buf[:bbox_size])
        buf = buf[bbox_size:]
        bbox = [box.x1, box.y1, box.x2, box.y2, box.score,
                box.track_id, box.track_score]
        bboxes.append(bbox)
    return bboxes

def single_video():
    video = '/bigdata/wcrichto/videos/meanGirls_short.mp4'
    db.ingest_video(('meangirls', video))
    sampler = db.sampler()
    tasks = sampler.all([('meangirls', 'meangirls_hist')])
    [table] = db.run(tasks, hist)

def video_collection():
    input_collection = db.ingest_video_collection(
        'meangirls',
        ['/n/scanner/wcrichto.new/videos/meanGirls_medium.mp4'],
        force=True)
    input_collection = db.collection('meangirls')
    sampler = db.sampler()
    strided = sampler.strided(input_collection, 1)
    output_collection = db.run(strided, hist, 'meangirls_hist', force=True)
    table = output_collection.tables(0)
    output_collection.profiler().write_trace('test.trace')



video_collection()
