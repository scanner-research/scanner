from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

util.download_video()

with Database() as db:
    blur_args = db.protobufs.BlurArgs()
    blur_args.kernel_size = 3;
    blur_args.sigma = 1;
    blur = db.ops.Blur(args=blur_args)

    video_paths = [util.download_video()]
    if not db.has_collection('example'):
        print('Ingesting videos into Scanner ...')
        collection, _ = db.ingest_video_collection('example', video_paths,
                                                   force=True)

    print('Running blur + encode...')
    in_collection = db.collection('example')
    collection = db.run(in_collection, blur,
                        'encode_example', force=True)
