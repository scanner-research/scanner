from scannerpy import Database, DeviceType, Job
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
        db, input_table, lambda t: t.all(),
        movie_name + '_bboxes')

    print('Drawing faces onto video...')
    frame = input_table.as_op().all()
    bboxes = bboxes_table.as_op().all()
    out_frame = db.ops.DrawBox(frame = frame, bboxes = bboxes)
    job = Job(columns = [out_frame], name = movie_name + '_bboxes_overlay')
    out_table = db.run(job, force=True)
    out_table.column('frame').save_mp4(movie_name + '_faces')

    print('Successfully generated {:s}_faces.mp4'.format(movie_name))
