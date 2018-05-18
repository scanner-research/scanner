import scannerpy
from scannerpy import Database, DeviceType, Job, FrameType
from scannerpy.stdlib import NetDescriptor, readers
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import util

from typing import Tuple

@scannerpy.register_python_op(name='PoseDraw')
def pose_draw(self, frame: FrameType, frame_poses: bytes) -> FrameType:
    for pose in readers.poses(frame_poses, self.protobufs):
        pose.draw(frame)
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

sampler = db.streams.All
sampler_args = {}

if db.has_gpu():
    print('Using GPUs')
    device = DeviceType.GPU
    pipeline_instances = -1
else:
    print('Using CPUs')
    device = DeviceType.CPU
    pipeline_instances = 1

frame = db.sources.FrameColumn()
poses_out = db.ops.OpenPose(
    frame=frame,
    pose_num_scales=3,
    pose_scale_gap=0.33,
    device=device)
drawn_frame = db.ops.PoseDraw(frame=frame, frame_poses=poses_out)
sampled_frames = sampler(drawn_frame)
output = db.sinks.Column(columns={'frame': sampled_frames})
job = Job(
    op_args={
        frame: input_table.column('frame'),
        sampled_frames: sampler_args,
        output: movie_name + '_drawn_poses',
})
[drawn_poses_table] = db.run(output=output, jobs=[job], work_packet_size=8, io_packet_size=64,
                             pipeline_instances_per_node=pipelineinstances,
                             force=True)

print('Writing output video...')
drawn_poses_table.column('frame').save_mp4('{:s}_poses'.format(movie_name))
