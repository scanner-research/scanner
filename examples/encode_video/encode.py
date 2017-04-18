from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import math
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util
from itertools import izip
import numpy as np

util.download_video()

with Database() as db:
    video_paths = [util.download_video()]
    #if not db.has_collection('example'):
    if True:
        print('Ingesting videos into Scanner ...')
        in_collection, _ = db.ingest_video_collection('example', video_paths,
                                                   force=True)

    print('Running blur + encode...')
    in_collection = db.collection('example')

    frame = in_collection.tables(0).as_op().all(item_size=250)
    print(frame)
    blur_frame = db.ops.Blur(frame = frame, kernel_size = 5, sigma = 1)
    #compressed_frame = blur_frame.compress(bitrate = 10 * 1024)
    #job = Job(columns = [compressed_frame], name = 'encode_example')
    job = Job(columns = [blur_frame], name = 'encode_example')
    blurred_table = db.run(job, True)

    frame = blurred_table.as_op().range(0, 100)
    flow_frame = db.ops.OpticalFlow(frame = frame)
    job = Job(columns = [flow_frame], name = 'encode_example2')
    flow_table = db.run(job, True)

    # orig_frames = [f[1] for f in in_collection.tables(0).columns('frame').load(rows=range(0, 100))]

    # blur_frames = [f[1] for f in blurred_table.columns('frame').load()]

    blurred_table.columns('frame').save_mp4('blur.mp4', fps=23.976)

    i = 0
    flow_frames = [f[1] for f in flow_table.columns('flow').load()]
    for f in flow_frames:
        cv2.imwrite('flow_left{:d}.png'.format(i), f[:,:,0] * 255)
        cv2.imwrite('flow_right{:d}.png'.format(i), f[:,:,1] * 255)
        i += 1

    #blur2_frames = [f[1] for f in blurred_table2.columns('frame').load()]

    # i = 0
    # for o, b1, b2 in izip(orig_frames, blur_frames, blur2_frames):
    #     f1 = o
    #     f2 = b1
    #     f3 = b2

    #     f = np.vstack((f1, f2, f3))
    #     cv2.imwrite('blur_{:04d}.png'.format(i), f)
    #     i += 1
