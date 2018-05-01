from scannerpy import Database, Job, ColumnType, DeviceType
import os
import sys
import math
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: {:s} path/to/your/video/file.mp4'.format(sys.argv[0]))
        sys.exit(1)

    weights_path = 'https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
    config_path = '/h/apoms/repos/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml'

    movie_path = sys.argv[1]
    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    sample_stride = 1

    db = Database()
    [input_table], failed = db.ingest_videos([('example', movie_path)],
                                             force=True)
    db.register_op('MaskRCNN',
                   [('frame', ColumnType.Video)],
                   [('vis_frame', ColumnType.Video)])
    kernel_path = script_dir + '/mask_rcnn_kernel.py'
    db.register_python_kernel('MaskRCNN', DeviceType.GPU, kernel_path)
    frame = db.sources.FrameColumn()
    strided_frame = frame.sample()

    # Call the newly created object detect op
    objdet_frame = db.ops.MaskRCNN(frame = strided_frame,
                                   config_path=config_path,
                                   weights_path=weights_path,
                                   device=DeviceType.GPU)

    output_op = db.sinks.Column(columns={'frame': objdet_frame})
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            strided_frame: db.sampler.strided(sample_stride),
            output_op: 'example_obj_detect',
        }
    )
    [out_table] = db.run(output=output_op, jobs=[job], force=True,
                         pipeline_instances_per_node=1)

    out_table.column('frame').save_mp4('{:s}_mrcnn_video')
    print('Successfully generated {:s}_mrcnn_video.mp4'.format(movie_name))
