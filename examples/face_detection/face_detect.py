from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import pipelines
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

movie_path = util.download_video() if len(sys.argv) <= 1 else sys.argv[1]
print('Detecting faces in movie {}'.format(movie_path))
movie_name = os.path.splitext(os.path.basename(movie_path))[0]

with Database() as db:
    print('Ingesting video into Scanner ...')
    [input_table], _ = db.ingest_videos(
        [(movie_name, movie_path)], force=True)

    print('Detecting faces...')
    bboxes_table = pipelines.detect_faces(
        db, [input_table.column('frame')], db.sampler.all(),
        movie_name + '_bboxes')[0]

    print('Drawing faces onto video...')
    frame = db.ops.FrameInput()
    bboxes = db.ops.Input()
    out_frame = db.ops.DrawBox(frame = frame, bboxes = bboxes)
    output = db.ops.Output(columns=[out_frame])
    job = Job(op_args={
        frame: input_table.column('frame'),
        bboxes: bboxes_table.column('bboxes'),
        output: movie_name + '_bboxes_overlay',
    })
    bulk_job = BulkJob(output=output, jobs=[job])
    [out_table] = db.run(bulk_job, force=True)
    out_table.column('frame').save_mp4(movie_name + '_faces')

    print('Successfully generated {:s}_faces.mp4'.format(movie_name))
