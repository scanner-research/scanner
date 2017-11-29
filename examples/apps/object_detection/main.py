from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import util
import numpy as np

util.download_video()

with Database() as db:
    video_path = util.download_video()
    if True or not db.has_table('example'):
        print('Ingesting video into Scanner ...')
        db.ingest_videos([('example', video_path)], force=True)

    input_table = db.table('example')

    descriptor = NetDescriptor.from_file(db, 'nets/faster_rcnn_coco.toml')
    caffe_args = db.protobufs.CaffeArgs()
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 1


    frame = db.ops.FrameInput()
    caffe_frame = db.ops.CaffeInput(
        frame = frame,
        args = caffe_args,
        device = DeviceType.GPU)
    cls_prob, rois, fc7 = db.ops.FasterRCNN(
        caffe_input = caffe_frame,
        args = caffe_args,
        device = DeviceType.GPU)
    bboxes, feature = db.ops.FasterRCNNOutput(
        cls_prob = cls_prob,
        rois = rois,
        fc7 = fc7,
        args = caffe_args,
        device = DeviceType.CPU)
    output = db.ops.Output(columns=[bboxes, feature])

    job = Job(op_args={
        frame: input_table.column('frame'),
        output: input_table.name() + '_detections'
    })
    bulk_job = BulkJob(output=output, jobs=[job])
    [output] = db.run(bulk_job, pipeline_instances_per_node = 1,
                      work_packet_size = 10, io_packet_size = 40, force=True)

    output = db.table(input_table.name() + '_detections')

    output.profiler().write_trace('detect_test.trace')

    print('Extracting frames...')

    def parse_features(buf, db):
        if len(buf) == 1:
            return np.zeros((1), dtype=np.dtype(np.float32))
        else:
            out = np.frombuffer(buf, dtype=np.dtype(np.int32))
            return out.reshape((-1, 4096))

    video_bboxes = [box for (_, box) in output.columns('bboxes').load(parsers.bboxes)]
    video_features = [feature for (_, feature) in output.columns('features').load(parse_features)]
    video_frames = [f[0] for _, f in db.table('example').load(['frame'], rows=range(800,1600))]

    print('Writing output video...')
    frame_shape = video_frames[0].shape
    print(frame_shape)
    output = cv2.VideoWriter(
        'example_detections.mkv',
        cv2.VideoWriter_fourcc(*'X264'),
       #cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        24.0,
        (frame_shape[1], frame_shape[0]))

    for (frame, bboxes) in zip(video_frames, video_bboxes):
        for bbox in bboxes:
            cv2.rectangle(
                frame,
                (int(bbox.x1), int(bbox.y1)),
                (int(bbox.x2), int(bbox.y2)),
                (255, 0, 0), 3)
        output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    output.release()
