from scannerpy import Database, DeviceType, Job, ColumnType, BulkJob
from scannerpy.stdlib import NetDescriptor, parsers, pipelines
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

script_dir = os.path.dirname(os.path.abspath(__file__))

movie_path = util.download_video() if len(sys.argv) <= 1 else sys.argv[1]
print('Detecting poses in video {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]

with Database() as db:
    video_path = movie_path
    if not db.has_table(movie_name):
        print('Ingesting video into Scanner ...')
        db.ingest_videos([(movie_name, video_path)], force=True)
    input_table = db.table(movie_name)

    sampler = db.sampler.range(120, 480)

    [poses_table] = pipelines.detect_poses(
        db, [input_table.column('frame')],
        sampler,
        '{:s}_poses'.format(movie_name))

    print('Drawing on frames...')
    db.register_op('PoseDraw', [('frame', ColumnType.Video), 'poses'],
                   [('frame', ColumnType.Video)])
    db.register_python_kernel('PoseDraw', DeviceType.CPU,
                              script_dir + '/pose_draw_kernel.py')
    frame = db.ops.FrameInput()
    sampled_frame = frame.sample()
    poses = db.ops.Input()
    drawn_frame = db.ops.PoseDraw(
        frame = sampled_frame,
        poses = poses)
    output = db.ops.Output(columns=[drawn_frame])
    job = Job(op_args={
        frame: input_table.column('frame'),
        sampled_frame: sampler,
        poses: poses_table.column('pose'),
        output: movie_name + '_drawn_poses',
    })
    bulk_job = BulkJob(output=output, jobs=[job])
    [drawn_poses_table] = db.run(bulk_job, force=True)
    print('Writing output video...')
    drawn_poses_table.column('frame').save_mp4('{:s}_poses'.format(
        movie_name))
