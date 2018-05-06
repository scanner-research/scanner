from scannerpy import Database, Job, ColumnType, DeviceType
import os
import sys
import math
import argparse
from tqdm import tqdm

import detectron_kernels

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--weights-path',
        type=str,
        required=True,
        help=('Path to the detectron model weights file. '
              'Can be a URL, in which case it will be cached after '
              'downloading.'))
    p.add_argument(
        '--config-path',
        type=str,
        required=True,
        help=('Path to the detectron model config yaml file.'))
    p.add_argument(
        '--video-path',
        type=str,
        required=True,
        help=('Path to video to process.'))

    args = p.parse_args()

    weights_path = args.weights_path
    config_path = args.config_path
    movie_path = args.video_path

    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    sample_stride = 1

    db = Database()
    [input_table], failed = db.ingest_videos(
        [('example', movie_path)], force=True)

    frame = db.sources.FrameColumn()
    strided_frame = frame.sample()

    # Call the newly created object detect op
    cls_boxes, cls_segms, cls_keyps = db.ops.Detectron(
        frame=strided_frame,
        config_path=config_path,
        weights_path=weights_path,
        device=DeviceType.GPU)

    objdet_frame = db.ops.DetectronVizualize(
        frame=strided_frame,
        cls_boxes=cls_boxes,
        cls_segms=cls_segms,
        cls_keyps=cls_keyps)

    output_op = db.sinks.Column(columns={'frame': objdet_frame})
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            strided_frame: db.sampler.strided(sample_stride),
            output_op: 'example_obj_detect',
        })
    [out_table] = db.run(
        output=output_op,
        jobs=[job],
        force=True,
        pipeline_instances_per_node=1)

    out_table.column('frame').save_mp4('{:s}_detected'.format(movie_name))
    print('Successfully generated {:s}_detected.mp4'.format(movie_name))
