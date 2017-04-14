from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import parsers
import os.path as osp
import numpy as np

db = Database()

if not db.has_table('example'):
    db.ingest_videos([('example', 'example.mp4')])
input_table = db.table('example')

frame, frame_info = input_table.as_op().all(warmup_size = 1)

flow = db.ops.OpticalFlow(
    frame = frame, frame_info = frame_info,
    device=DeviceType.GPU)

job = Job(columns = [flow, frame_info], name = 'example_flows')

output_table = db.run(job)

vid_flows = [flow for _, flow in output_table.load(['flow', 'frame_info'], parsers.flow)]
np.save('flows.npy', vid_flows)
