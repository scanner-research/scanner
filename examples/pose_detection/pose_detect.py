from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import NetDescriptor, parsers, pipelines
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
    video_path = util.download_video()
    if not db.has_table('example'):
        print('Ingesting video into Scanner ...')
        db.ingest_videos([('example', video_path)], force=True)
    input_table = db.table('example')

    poses_table = pipelines.detect_poses(
        db, [input_table], lambda t: t.range(0, 100, task_size = 25),
        'example_poses',
        height = 360)[0]

    print('Extracting frames...')
    video_poses = [pose for (_, pose) in poses_table.columns('poses').load(parsers.poses)]
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
            for i in range(18):
                if pose[i, 2] < 0.25: continue
                cv2.circle(
                    frame,
                    (int(pose[i, 1]), int(pose[i, 0])),
                    8,
                    (255, 0, 0), 3)
        output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
