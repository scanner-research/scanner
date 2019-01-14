from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import pipelines
import subprocess
import cv2
import sys
import os.path
import scipy.misc
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
import util

def usage():
    print('Usage: main.py <video_file_or_image_file>')

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        usage()
        exit(1)

    db = Database()
    if os.path.isdir(sys.argv[1]):
        is_movie = False
        dir_name = sys.argv[1]
        image_paths = os.listdir(dir_name)
        image_paths = [os.path.join(dir_name, p) for p in image_paths]
        print('Ingesting images into Scanner ...')
        [input_table] = pipelines.ingest_images(db, image_paths, dir_name, force=True)
        table_name = dir_name
    elif os.path.isfile(sys.argv[1]):
        is_movie = True
        movie_path = sys.argv[1]
        print('Detecting faces in movie {}'.format(movie_path))
        movie_name = os.path.splitext(os.path.basename(movie_path))[0]
    
        print('Ingesting video into Scanner ...')
        [input_table], _ = db.ingest_videos(
            [(movie_name, movie_path)], force=True)
        table_name = movie_name
    else:
        usage()
        exit(1)
    
    sampler = db.streams.All
    sampler_args = {}
    
    print('Detecting faces...')
    [bboxes_table] = pipelines.detect_faces(
        db, [input_table.column('frame')],
        sampler,
        sampler_args,
        table_name + '_faces')
    
    print('Drawing faces onto frames...')
    frame = db.sources.FrameColumn()
    sampled_frame = sampler(frame)
    bboxes = db.sources.Column()
    out_frame = db.ops.DrawBox(frame=sampled_frame, bboxes=bboxes)
    output = db.sinks.Column(columns={'frame': out_frame})
    job = Job(op_args={
        frame: input_table.column('frame'),
        sampled_frame: sampler_args,
        bboxes: bboxes_table.column('bboxes'),
        output: table_name + '_faces_overlay',
    })
    [out_table] = db.run(output=output, jobs=[job], force=True)
    
    if not is_movie:
        out_dir = dir_name + '_faces'
        os.makedirs(out_dir, exist_ok=True)
        for i, img in enumerate(out_table.column('frame').load()):
            path = image_paths[i]
            file_name = os.path.basename(path)
            scipy.misc.imsave(os.path.join(out_dir, file_name), img)
        print('Successfully generated {:s}_faces'.format(dir_name))
    else:
        out_table.column('frame').save_mp4(movie_name + '_faces')
        print('Successfully generated {:s}_faces.mp4'.format(movie_name))

