from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import parsers
import os
import os.path as osp
import numpy as np
import time
import sys

if len(sys.argv) <= 1:
    print('Usage: main.py <video_file>')
    exit(1)

video_path = sys.argv[1]
print('Performing optical flow on {}...'.format(video_path))
video_name = os.path.splitext(os.path.basename(video_path))[0]

with Database() as db:
    if not db.has_table(video_name):
        db.ingest_videos([(video_name, video_path)])
    input_table = db.table(video_name)

    sampler = db.sampler.all()

    frame = db.ops.FrameInput()
    flow = db.ops.OpticalFlow(
        frame = frame,
        device=DeviceType.CPU)
    sampled_flow = flow.sample()
    output = db.ops.Output(columns=[sampled_flow])

    job = Job(op_args={
        frame: input_table.column('frame'),
        sampled_flow: sampler,
        output: input_table.name() + '_flow'
    })
    bulk_job = BulkJob(output=output, jobs=[job])

    [output_table] = db.run(bulk_job, pipeline_instances_per_node=1, force=True)

    vid_flows = [flow[0] for _, flow in output_table.load(['flow'], rows=[0])]
    np.save('flows.npy', vid_flows)
