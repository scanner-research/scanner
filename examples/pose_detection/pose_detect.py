from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

util.download_video()

with Database() as db:

    descriptor = NetDescriptor.from_file(db, 'nets/cpm2.toml')
    cpm2_args = db.protobufs.CPM2Args()
    cpm2_args.scale = 368.0/720.0
    caffe_args = cpm2_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 1

    video_path = util.download_video()
    if not db.has_table('example'):
        print('Ingesting video into Scanner ...')
        db.ingest_videos([('example', video_path)], force=True)
    input_table = db.table('example')

    frame, frame_info = input_table.as_op().all(item_size = 50)
    cpm2_input = db.ops.CPM2Input(
        frame = frame, frame_info = frame_info,
        args = cpm2_args,
        device = DeviceType.GPU)
    cpm2_resized_map, cpm2_joints = db.ops.CPM2(
        cpm2_input = cpm2_input,
        frame_info = frame_info,
        args = cpm2_args,
        device = DeviceType.GPU)
    poses = db.ops.CPM2Output(
        cpm2_resized_map = cpm2_resized_map,
        cpm2_joints = cpm2_joints,
        frame_info = frame_info,
        args = cpm2_args)

    job = Job(columns = [poses], name = 'example_poses')
    output = db.run(job, True)

    print('Extracting frames...')
    video_poses = [pose for (_, pose) in output.columns('poses').load(parsers.poses)]
    video_frames = [f[0] for _, f in db.table('example').load(['frame'])]

    print('Writing output video...')
    frame_shape = video_frames[0].shape
    output = cv2.VideoWriter(
        'example_poses.mkv',
        cv2.VideoWriter_fourcc(*'X264'),
        24.0,
        (frame_shape[1], frame_shape[0]))

    for (frame, frame_poses) in zip(video_frames, video_poses):
        for pose in frame_poses:
            for i in range(15):
                if pose[i, 2] < 0.1: continue
                cv2.circle(
                    frame,
                    (int(pose[i, 1]), int(pose[i, 0])),
                    8,
                    (255, 0, 0), 3)
        output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
