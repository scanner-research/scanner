from scannerpy import Client, DeviceType
from scannerpy.storage import NamedVideoStream, PythonStream
import os
import sys
import math
import numpy as np
from tqdm import tqdm
import six.moves.urllib as urllib

import kernels

# What model to download.
MODEL_TEMPLATE_URL = 'http://download.tensorflow.org/models/object_detection/{:s}.tar.gz'

def main():
    if len(sys.argv) <= 1:
        print('Usage: {:s} path/to/your/video/file.mp4'.format(sys.argv[0]))
        sys.exit(1)

    movie_path = sys.argv[1]
    print('Detecting objects in movie {}'.format(movie_path))
    movie_name = os.path.splitext(os.path.basename(movie_path))[0]

    sc = Client()

    stride = 1
    input_stream = NamedVideoStream(sc, movie_name, path=movie_path)
    frame = sc.io.Input([input_stream])
    strided_frame = sc.streams.Stride(frame, [stride])

    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    model_url = MODEL_TEMPLATE_URL.format(model_name)
    objdet_frame = sc.ops.ObjDetect(
        frame=strided_frame,
        dnn_url=model_url,
        device=DeviceType.GPU if sc.has_gpu() else DeviceType.CPU,
        batch=2)

    detect_stream = NamedVideoStream(sc, movie_name + '_detect')
    output_op = sc.io.Output(objdet_frame, [detect_stream])
    sc.run(output_op)

    print('Extracting data from Scanner output...')
    # bundled_data_list is a list of bundled_data
    # bundled data format: [box position(x1 y1 x2 y2), box class, box score]
    bundled_data_list = list(tqdm(detect_stream.load()))
    print('Successfully extracted data from Scanner output!')

    # run non-maximum suppression
    bundled_np_list = kernels.nms_bulk(bundled_data_list)
    bundled_np_list = kernels.smooth_box(bundled_np_list, min_score_thresh=0.5)

    print('Writing frames to {:s}_obj_detect.mp4'.format(movie_name))

    frame = sc.io.Input([input_stream])
    bundled_data = sc.io.Input([PythonStream(bundled_np_list)])
    strided_frame = sc.streams.Stride(frame, [stride])
    drawn_frame = sc.ops.TFDrawBoxes(frame=strided_frame,
                                     bundled_data=bundled_data,
                                     min_score_thresh=0.5)
    drawn_stream = NamedVideoStream(sc, movie_name + '_drawn_frames')
    output_op = sc.io.Output(drawn_frame, [drawn_stream])
    sc.run(output_op)

    drawn_stream.save_mp4(movie_name + '_obj_detect')

    input_stream.delete(sc)
    detect_stream.delete(sc)
    drawn_stream.delete(sc)

    print('Successfully generated {:s}_obj_detect.mp4'.format(movie_name))


if __name__ == '__main__':
    main()


