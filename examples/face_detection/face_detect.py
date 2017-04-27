from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import NetDescriptor, parsers, bboxes, writers
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

with Database() as db:
    # TODO(wcrichto): comment the demo. Make the Scanner philosophy more clear.
    # Add some figures to the wiki perhaps explaining the high level

    descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    print('Ingesting video into Scanner ...')
    [input_table], _ = db.ingest_videos([('example', util.download_video())], force=True)
    base_batch = 4
    base_size = 1280*720
    # TODO(apoms): determine automatically from video
    current_size = 640*480
    current_batch = math.floor(base_size / float(current_size) * base_batch)

    print('Running face detector...')
    outputs = []
    scales = [1.0, 0.5, 0.25, 0.125]
    batch_sizes = [int(current_batch * (2**i))
                   for i in range(len(scales))]
    for scale, batch in zip(scales, batch_sizes):
        print('Scale {}...'.format(scale))
        facenet_args.scale = scale
        caffe_args.batch_size = batch
        frame = input_table.as_op().all()
        frame_info = db.ops.InfoFromFrame(frame = frame)
        facenet_input = db.ops.FacenetInput(
            frame = frame,
            args = facenet_args,
            device = DeviceType.GPU)
        facenet = db.ops.Facenet(
            facenet_input = facenet_input,
            args = facenet_args,
            device = DeviceType.GPU)
        facenet_output = db.ops.FacenetOutput(
            facenet_output = facenet,
            original_frame_info = frame_info,
            args = facenet_args)

        job = Job(
            columns = [facenet_output],
            name = 'example_faces_{}'.format(scale))
        output = db.run(job, force=True)
        outputs.append(output)

    all_bboxes = [
        [box for (_, box) in out.load(['bboxes'], parsers.bboxes)]
        for out in outputs]

    nms_bboxes = []
    frames = len(all_bboxes[0])
    runs = len(all_bboxes)
    for fi in range(frames):
        frame_bboxes = []
        for r in range(runs):
            frame_bboxes += (all_bboxes[r][fi])
        frame_bboxes = bboxes.nms(frame_bboxes, 0.3)
        nms_bboxes.append(frame_bboxes)

    bboxes_table = db.new_table(
        'example_bboxes',
        ['bboxes'],
        [[bb] for bb in nms_bboxes],
        writers.bboxes,
        force=True)

    frame = input_table.as_op().all()
    bboxes = bboxes_table.as_op().all()
    out_frame = db.ops.DrawBox(frame = frame, bboxes = bboxes)
    job = Job(columns = [out_frame], name = 'example_bboxes_overlay')
    out_table = db.run(job, force=True)
    out_table.column('frame').save_mp4('example_faces')
