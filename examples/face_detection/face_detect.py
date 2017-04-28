from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import pipelines
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

with Database() as db:
    print('Ingesting video into Scanner ...')
    [input_table], _ = db.ingest_videos([('example', util.download_video())], force=True)

    print('Detecting faces...')
    bboxes_table = pipelines.detect_faces(
        db, lambda: input_table.as_op().all(), 'example_bboxes')

    print('Drawing faces onto video...')
    frame = input_table.as_op().all()
    bboxes = bboxes_table.as_op().all()
    out_frame = db.ops.DrawBox(frame = frame, bboxes = bboxes)
    job = Job(columns = [out_frame], name = 'example_bboxes_overlay')
    out_table = db.run(job, force=True)
    out_table.column('frame').save_mp4('example_faces')

    print('Successfully generated example_faces.mp4')
