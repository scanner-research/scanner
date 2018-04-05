from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import NetDescriptor
import numpy as np
import cv2
import struct
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

video_path = util.download_video() if len(sys.argv) <= 1 else sys.argv[1]
print('Performing classification on video {}'.format(video_path))
video_name = os.path.splitext(os.path.basename(video_path))[0]

with Database() as db:
    [input_table], _ = db.ingest_videos(
        [(video_name, video_path)], force=True)

    descriptor = NetDescriptor.from_file(db, 'nets/resnet.toml')

    batch_size = 48
    frame = db.sources.FrameColumn()
    caffe_input = db.ops.CaffeInput(
        frame = frame,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        device=DeviceType.GPU)
    caffe_output = db.ops.Caffe(
        caffe_frame = caffe_input,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        batch = batch_size,
        device=DeviceType.GPU)
    output = db.sinks.Column(columns={'softmax': caffe_output})

    job = Job(op_args={
        frame: input_table.column('frame'),
        output: input_table.name() + '_classification'
    })

    [output] = db.run(output=output, jobs=[job], pipeline_instances_per_node=1)
