import scannerpy
from scannerpy import Database, DeviceType, Job, FrameType
from scannerpy.stdlib import NetDescriptor, readers, pipelines
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import util

from typing import Tuple

@scannerpy.register_python_op()
class PoseDraw(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, frame: FrameType, frame_poses: bytes) -> FrameType:
        for all_pose in readers.poses(frame_poses, self.protobufs):
            pose = all_pose.pose_keypoints()
            for i in range(18):
                if pose[i, 2] < 0.35: continue
                x = int(pose[i, 0] * frame.shape[1])
                y = int(pose[i, 1] * frame.shape[0])
                cv2.circle(frame, (x, y), 8, (255, 0, 0), 3)
        return frame


if len(sys.argv) <= 1:
    print('Usage: main.py <video_file>')
    exit(1)

movie_path = sys.argv[1]
print('Detecting poses in video {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]

db = Database()
video_path = movie_path
if not db.has_table(movie_name):
    print('Ingesting video into Scanner ...')
    db.ingest_videos([(movie_name, video_path)], force=True)
input_table = db.table(movie_name)

sampler = db.streams.Range
sampler_args = {'start': 120, 'end': 480}

[poses_table] = pipelines.detect_poses(db, [input_table.column('frame')],
                                       sampler,
                                       sampler_args,
                                       '{:s}_poses'.format(movie_name))

print('Drawing on frames...')
frame = db.sources.FrameColumn()
sampled_frame = sampler(frame)
poses = db.sources.Column()
drawn_frame = db.ops.PoseDraw(frame=sampled_frame, poses=poses)
output = db.sinks.Column(columns={'frame': drawn_frame})
job = Job(
    op_args={
        frame: input_table.column('frame'),
        sampled_frame: sampler_args,
        poses: poses_table.column('poses'),
        output: movie_name + '_drawn_poses',
    })
[drawn_poses_table] = db.run(output=output, jobs=[job], force=True)
print('Writing output video...')
drawn_poses_table.column('frame').save_mp4('{:s}_poses'.format(movie_name))
