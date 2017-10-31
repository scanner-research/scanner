from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import parsers
import os.path as osp
import numpy as np
import time

with Database() as db:
    if not db.has_table('example'):
        db.ingest_videos([('example', '/tmp/example.mp4')])
    input_table = db.table('example')

    frame = db.ops.FrameInput()
    flow = db.ops.OpticalFlow(
        frame = frame,
        device=DeviceType.CPU)
    sampled_flow = flow.sample()
    output = db.ops.Output(columns=[sampled_flow])

    job = Job(op_args={
        frame: input_table.column('frame'),
        sampled_flow: db.sampler.range(0, 20),
        output: input_table.name() + '_flow'
    })
    bulk_job = BulkJob(output=output, jobs=[job])

    [output_table] = db.run(bulk_job, pipeline_instances_per_node=1, force=True)

    vid_flows = [flow[0] for _, flow in output_table.load(['flow'], rows=[0])]
    np.save('flows.npy', vid_flows)
