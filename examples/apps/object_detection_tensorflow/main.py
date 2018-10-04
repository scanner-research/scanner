from scannerpy import Database, Job, ColumnType, DeviceType
import os
import sys
import math
import numpy as np
from tqdm import tqdm
import six.moves.urllib as urllib

import kernels

# What model to download.
MODEL_TEMPLATE_URL = 'http://download.tensorflow.org/models/object_detection/{:s}.tar.gz'

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: {:s} path/to/your/video/file.mp4'.format(sys.argv[0]))
        sys.exit(1)

    movie_path = sys.argv[1]
    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    db = Database()

    [input_table], failed = db.ingest_videos([('example', movie_path)], force=True)

    stride = 1

    frame = db.sources.FrameColumn()
    strided_frame = db.streams.Stride(frame, stride)

    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    model_url = MODEL_TEMPLATE_URL.format(model_name)

    # Call the newly created object detect op
    objdet_frame = db.ops.ObjDetect(
        frame=strided_frame,
        dnn_url=model_url,
        device=DeviceType.GPU if db.has_gpu() else DeviceType.CPU,
        batch=2)

    output_op = db.sinks.Column(columns={'bundled_data': objdet_frame})
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            output_op: 'example_obj_detect',
        })

    [out_table] = db.run(output=output_op, jobs=[job], force=True,
                         pipeline_instances_per_node=1)

    out_table.profiler().write_trace('obj.trace')

    print('Extracting data from Scanner output...')

    # bundled_data_list is a list of bundled_data
    # bundled data format: [box position(x1 y1 x2 y2), box class, box score]
    bundled_data_list = [
        np.fromstring(box, dtype=np.float32)
        for box in tqdm(out_table.column('bundled_data').load())
    ]
    print('Successfully extracted data from Scanner output!')

    # run non-maximum suppression
    bundled_np_list = kernels.nms_bulk(bundled_data_list)
    bundled_np_list = kernels.smooth_box(bundled_np_list, min_score_thresh=0.5)

    print('Writing frames to {:s}_obj_detect.mp4'.format(movie_name))

    frame = db.sources.FrameColumn()
    bundled_data = db.sources.Python()
    strided_frame = db.streams.Stride(frame, stride)
    drawn_frame = db.ops.TFDrawBoxes(frame=strided_frame,
                                     bundled_data=bundled_data,
                                     min_score_thresh=0.5)
    output_op = db.sinks.Column(columns={'frame': drawn_frame})
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            bundled_data: {'data': pickle.dumps(bundled_np_list)},
            output_op: 'example_drawn_frames',
        })

    [out_table] = db.run(output=output_op, jobs=[job], force=True,
                         pipeline_instances_per_node=1)

    out_table.column('frame').save_mp4(movie_name + '_obj_detect')

    print('Successfully generated {:s}_obj_detect.mp4'.format(movie_name))
