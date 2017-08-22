from scannerpy import Database, DeviceType, Job, ColumnType
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
        height = 720)[0]

    print('Drawing on frames...')
    db.register_op('PoseDraw', [('frame', ColumnType.Video), 'poses'],
                   [('frame', ColumnType.Video)])
    db.register_python_kernel('PoseDraw', DeviceType.CPU,
                              script_dir + '/pose_draw_kernel.py')
    drawn_frames = db.ops.PoseDraw(
        frame = input_table.as_op().range(0, num_frames, task_size = 100),
        poses = poses_table.as_op().all(task_size = 100))
    job = Job(columns = [drawn_frames], name = '720_drawn_poses')
    drawn_poses_table = db.run(job, force=True)
    print('Writing output video...')
    drawn_poses_table.column('frame').save_mp4('example_poses')
