from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
import os.path as osp
import numpy as np

db = Database()

input = db.ops.Input()
flow = db.ops.OpticalFlow(
    inputs=[(input,['frame', 'frame_info'])],
    device=DeviceType.GPU)
output = db.ops.Output(inputs=[(flow, ['flow']), (input, ['frame_info'])])

if not db.has_table('example'):
    db.ingest_videos([('example', 'example.mp4')])
input_table = db.table('example')

tasks = db.sampler().all([(input_table.name(), 'example_flows')], warmup_size=1)
[output_table] = db.run(tasks, output)

vid_flows = [flow for _, flow in output_table.load((0, 1), parsers.flow)]
np.save('flows.npy', vid_flows)
