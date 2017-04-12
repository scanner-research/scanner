from scannerpy import Database, DeviceType
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

    # TODO(wcrichto): comment the demo. Make the Scanner philosophy more clear.
    # Add some figures to the wiki perhaps explaining the high level

    descriptor = NetDescriptor.from_file(db, 'nets/cpm2.toml')
    cpm2_args = db.protobufs.CPM2Args()
    cpm2_args.scale = 368.0/720.0
    caffe_args = cpm2_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 1


    input_op = db.ops.Input()
    cpm2_input = db.ops.CPM2Input(
        inputs=[(input_op, ["frame", "frame_info"])],
        args=cpm2_args,
        device=DeviceType.GPU)
    cpm2 = db.ops.CPM2(
        inputs=[(cpm2_input, ["cpm2_input"]), (input_op, ["frame_info"])],
        args=cpm2_args,
        device=DeviceType.GPU)
    cpm2_output = db.ops.CPM2Output(
        inputs=[(cpm2, ["cpm2_resized_map", "cpm2_joints"]),
                (input_op, ["frame_info"])],
        args=cpm2_args)

    video_path = util.download_video()
    if True or not db.has_table('example'):
        print('Ingesting video into Scanner ...')
        db.ingest_videos([('example', video_path)], force=True)

    sampler = db.sampler()
    print('Running pose detector...')
    outputs = []

    tasks = sampler.all([('example', 'example_poses')], item_size=50)
    [output] = db.run(tasks, cpm2_output, force=True)

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
