from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, loaders
from functools import partial
import os
import subprocess
import cv2

db = Database()

descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
facenet_args = db.protobufs.FacenetArgs()
facenet_args.scale = 0.5
facenet_args.threshold = 0.5
caffe_args = facenet_args.caffe_args
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 5

table_input = db.ops.Input()
caffe_input = db.ops.FacenetInput(
    inputs=[(table_input, ["frame", "frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
caffe = db.ops.Facenet(
    inputs=[(caffe_input, ["caffe_frame"]), (table_input, ["frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
caffe_output = db.ops.FacenetOutput(
    inputs=[(caffe, ["caffe_output"]), (table_input, ["frame_info"])],
    args=facenet_args)

if not os.path.isfile('example.mp4'):
    print('Downloading video...')
    subprocess.check_call(
        'youtube-dl -f mp4 -o "example.%(ext)s" '
        '"https://www.youtube.com/watch?v=dQw4w9WgXcQ"',
        shell=True)

if not db.has_table('example'):
    print('Ingesting video into Scanner ...')
    db.ingest_videos([('example', 'example.mp4')])

sampler = db.sampler()
tasks = sampler.all([('example', 'example_faces')])
print('Running face detector...')
[faces_table] = db.run(tasks, caffe_output, force=True, io_item_size=200, work_item_size=50)

print('Extracting frames...')
video_faces = [f for _, f in faces_table.columns(0).load(partial(loaders.bboxes, db))]
video_frames = [f for _, f in db.table('example').columns(0).load()]

print('Writing output video...')
frame_shape = video_frames[0].shape
output = cv2.VideoWriter(
    'example_faces.mkv',
    cv2.VideoWriter_fourcc(*'X264'),
    24.0,
    (frame_shape[1], frame_shape[0]))

for (frame, frame_faces) in zip(video_frames, video_faces):
    for face in frame_faces:
        if face[4] < 0.5: continue
        face = map(int, face)
        cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (255, 0, 0), 3)
    output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
