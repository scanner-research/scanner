from scannerpy import Database, DeviceType, NetDescriptor
from functools import partial
import numpy as np
import cv2
import struct

db = Database()

descriptor = NetDescriptor.from_file(db, 'features/caffe_facenet.toml')
facenet_args = db.protobufs.FacenetArgs()
facenet_args.scale = 1.0
facenet_args.threshold = 0.5
caffe_args = facenet_args.caffe_args
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 96

table_input = db.ops.Input()
caffe_input = db.ops.FacenetInput(
    inputs=[(table_input, ["frame", "frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
caffe = db.ops.Facenet(
    inputs=[(caffe_input, ["caffe_frame"]), (table_input, ["frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
caffe_output = db.ops.FacenetOutput(
    inputs=[(caffe, ["caffe_output"]), (table_input, ["frame_info"])],
    args=facenet_args)

input_collection = db.ingest_video_collection('test', ['test.mp4'])
output_collection = db.run(input_collection, caffe_output, 'test_faces')

def parse_bboxes(db, buf):
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

bboxes = output_collection.tables(0).columns(0).load(parse_bboxes)
