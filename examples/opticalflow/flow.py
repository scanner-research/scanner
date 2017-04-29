from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import parsers
import os.path as osp
import numpy as np
import time

with Database() as db:
    if not db.has_table('example'):
        db.ingest_videos([('example', '/tmp/example.mp4')])
    input_table = db.table('example')

    frame = input_table.as_op().range(0, 20, item_size=10)
    flow = db.ops.OpticalFlow(
        frame = frame,
        device=DeviceType.CPU)
    job = Job(columns = [flow], name = 'example_flows')

    output_table = db.run(job, force=True)

    vid_flows = [flow[0] for _, flow in output_table.load(['flow'], rows=[0])]
    np.save('flows.npy', vid_flows)
