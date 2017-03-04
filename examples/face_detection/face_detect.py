from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import os
import subprocess
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

db = Database()

# TODO(wcrichto): comment the demo. Make the Scanner philosophy more clear.
# Add some figures to the wiki perhaps explaining the high level

descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
facenet_args = db.protobufs.FacenetArgs()
facenet_args.threshold = 0.5
caffe_args = facenet_args.caffe_args
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 5

table_input = db.ops.Input()
facenet_input = db.ops.FacenetInput(
    inputs=[(table_input, ["frame", "frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
facenet = db.ops.Facenet(
    inputs=[(facenet_input, ["facenet_input"]), (table_input, ["frame_info"])],
    args=facenet_args,
    device=DeviceType.GPU)
facenet_output = db.ops.FacenetOutput(
    inputs=[(facenet, ["facenet_output"]), (table_input, ["frame_info"])],
    args=facenet_args)

if not db.has_table('example'):
    print('Ingesting video into Scanner ...')
    db.ingest_videos([('example', util.download_video())], force=True)

sampler = db.sampler()
print('Running face detector...')
outputs = []
for scale in [0.125, 0.25, 0.5, 1.0]:
    print('Scale {}...'.format(scale))
    facenet_args.scale = scale
    tasks = sampler.all([('example', 'example_faces_{}'.format(scale))],
                        item_size=50)
    [output] = db.run(tasks, facenet_output, force=True, work_item_size=5)
    outputs.append(output)

all_bboxes = [
    [box for (_, box) in out.load([0], parsers.bboxes)]
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
video_frames = [f[0] for _, f in db.table('example').load([0])]

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
