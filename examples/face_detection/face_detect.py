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

util.download_video()

with Database(master='crissy.pdl.local.cmu.edu:5001',
              workers=['crissy.pdl.local.cmu.edu:5002',
                       'pismo.pdl.local.cmu.edu:5002']) as db:

    # TODO(wcrichto): comment the demo. Make the Scanner philosophy more clear.
    # Add some figures to the wiki perhaps explaining the high level

    descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 2

    print('Ingesting video into Scanner ...')
    [input_table], _ = db.ingest_videos([('example', util.download_video())], force=True)
    base_batch = 4
    base_size = 1280*720
    # TODO(apoms): determine automatically from video
    current_size = 1280*720
    current_batch = math.floor(base_size / float(current_size) * base_batch)

    print('Running face detector...')
    outputs = []
    scales = [0.125, 0.25, 0.5, 1.0]
    batch_sizes = [int(current_batch * (2**i)) for i in range(len(scales))]
    batch_sizes.reverse()
    for scale, batch in zip(scales, batch_sizes):
        print('Scale {}...'.format(scale))
        facenet_args.scale = scale
        caffe_args.batch_size = batch
        frame, frame_info = input_table.as_op().all(item_size = 50)

        facenet_input = db.ops.FacenetInput(
            frame = frame, frame_info = frame_info,
            args = facenet_args,
            device = DeviceType.GPU)
        facenet = db.ops.Facenet(
            facenet_input = facenet_input,
            frame_info = frame_info,
            args = facenet_args,
            device = DeviceType.GPU)
        facenet_output = db.ops.FacenetOutput(
            facenet_output = facenet,
            frame_info = frame_info,
            args = facenet_args)

        job = Job(columns = [facenet_output], name = 'example_faces_{}'.format(scale))

        output = db.run(job, force=True, work_item_size=5)
        outputs.append(output)

    all_bboxes = [
        [box for (_, box) in out.load(['bboxes'], parsers.bboxes)]
        for out in outputs]

    nms_bboxes = []
    frames = len(all_bboxes[0])
    runs = len(all_bboxes)
    for fi in range(frames):
        frame_bboxes = []
        for r in range(runs):
            frame_bboxes += (all_bboxes[r][fi])
        frame_bboxes = bboxes.nms(frame_bboxes, 0.3)
        nms_bboxes.append(frame_bboxes)

    print('Extracting frames...')
    video_faces = nms_bboxes
    video_frames = [f[0] for _, f in db.table('example').load(['frame'])]

    print('Writing output video...')
    frame_shape = video_frames[0].shape
    output = cv2.VideoWriter(
        'example_faces.mkv',
        cv2.VideoWriter_fourcc(*'X264'),
        24.0,
        (frame_shape[1], frame_shape[0]))

    for (frame, frame_faces) in zip(video_frames, video_faces):
        for face in frame_faces:
            if face.score < 0.5: continue
            cv2.rectangle(
                frame,
                (int(face.x1), int(face.y1)),
                (int(face.x2), int(face.y2)),
                (255, 0, 0), 3)
        output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
