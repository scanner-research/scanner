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

<<<<<<< Updated upstream
    video_path = util.download_video()
    if not db.has_table('example'):
        print('Ingesting video into Scanner ...')
        db.ingest_videos([('example', video_path)], force=True)
    input_table = db.table('example')
=======

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

    video_paths = [
        '/n/scanner/datasets/kcam/20150207_154033_788.mp4',
        '/n/scanner/datasets/kcam/20150307_120624_341.mp4',
        '/n/scanner/datasets/kcam/20141028_151111_219.mp4',
        '/n/scanner/datasets/kcam/20150101_130918_979.mp4'
    ]
    if False and not db.has_table('example'):
        print('Ingesting video into Scanner ...')
        collection, _ = db.ingest_video_collection('kcam_dfouhey', video_paths, force=True)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    job = Job(columns = [poses], name = 'example_poses')
    output = db.run(job, True)
=======
    #tasks = sampler.all(collection, item_size=50)
    #collection = db.run(tasks, cpm2_output, 'kcam_dfouhey_poses', force=True)
    collection = db.collection('kcam_dfouhey_poses')
    for path, table in zip(video_paths, collection.tables()):
        poses = [pose for (_, pose) in table.columns('poses').load(parsers.poses)]
        with open(os.path.splitext(os.path.basename(path))[0], 'w') as f:
            for list_pose in poses:
                for p in list_pose:
                    for i in range(15):
                        f.write('{:f} {:f} {:f} '.format(p[i, 1],
                                                         p[i, 0],
                                                         p[i, 2]))
                    f.seek(-1, 1)
                    f.write(', ')
                if len(list_pose) > 0:
                    f.seek(-1, 1)
                f.write('\n')

    exit(1)
>>>>>>> Stashed changes

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
