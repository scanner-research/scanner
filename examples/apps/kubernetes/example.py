from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import readers

import numpy as np
import cv2
import sys
import os.path
import subprocess as sp

print('Finding master IP...')
ip = sp.check_output(
    '''
kubectl get pods -l 'app=scanner-master' -o json | \
jq '.items[0].spec.nodeName' -r | \
xargs -I {} kubectl get nodes/{} -o json | \
jq '.status.addresses[] | select(.type == "ExternalIP") | .address' -r
''',
    shell=True).strip().decode('utf-8')

port = sp.check_output(
    '''
kubectl get svc/scanner-master -o json | \
jq '.spec.ports[0].nodePort' -r
''',
    shell=True).strip().decode('utf-8')

master = '{}:{}'.format(ip, port)

print('Connecting to Scanner database...')
db = Database(master=master, start_cluster=False)

print('Running Scanner job...')
example_video_path = 'sample.mp4'
[input_table], _ = db.ingest_videos(
    [('example', example_video_path)], force=True, inplace=True)

print(db.summarize())

frame = db.sources.FrameColumn()
hist = db.ops.Histogram(frame=frame)
output_op = db.sinks.Column(columns={'hist': hist})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_hist'
})

output_tables = db.run(output=output_op, jobs=[job], force=True)

video_hists = output_tables[0].column('hist').load(readers.histograms)

num_rows = 0
for frame_hists in video_hists:
    assert len(frame_hists) == 3
    assert frame_hists[0].shape[0] == 16
    num_rows += 1
assert num_rows == db.table('example').num_rows()

print('Complete!')
