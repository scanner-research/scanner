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
    kubectl get services scanner-master --output json | jq -r '.status.loadBalancer.ingress[0].hostname'
    ''',
    shell=True).strip().decode('utf-8')

port = sp.check_output(
    '''
kubectl get svc/scanner-master -o json | \
jq '.spec.ports[0].port' -r
''',
    shell=True).strip().decode('utf-8')

master = '{}:{}'.format(ip, port)
print('Master ip: {:s}'.format(master))

with open('config.toml', 'w') as f:
    config_text = sp.check_output(
        '''
        kubectl get configmaps scanner-configmap -o json | \
        jq '.data["config.toml"]' -r
        ''',
        shell=True).strip().decode('utf-8')
    f.write(config_text)

print('Connecting to Scanner database...')
db = Database(
    master=master,
    start_cluster=False,
    config_path='./config.toml',
    grpc_timeout=120)

print('Running Scanner job...')
example_video_path = 'sample.mp4'
[input_table], _ = db.ingest_videos(
    [('example', example_video_path)], force=True, inplace=True)

print(db.summarize())

frame = db.sources.FrameColumn()
r_frame = db.ops.Resize(frame=frame, width=320, height=240)
output_op = db.sinks.Column(columns={'frame': r_frame})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_frame'
})

output_tables = db.run(output=output_op, jobs=[job], force=True)

output_tables[0].column('frame').save_mp4('resized_video')

print('Complete!')
