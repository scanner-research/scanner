import scannerpy as sp
import scannertools.imgproc

import numpy as np
import cv2
import sys
import os.path
import subprocess as sub

print('Finding master IP...')
ip = sub.check_output(
    '''
kubectl get pods -l 'app=scanner-master' -o json | \
jq '.items[0].spec.nodeName' -r | \
xargs -I {} kubectl get nodes/{} -o json | \
jq '.status.addresses[] | select(.type == "ExternalIP") | .address' -r
''',
    shell=True).strip().decode('utf-8')

port = sub.check_output(
    '''
kubectl get svc/scanner-master -o json | \
jq '.spec.ports[0].nodePort' -r
''',
    shell=True).strip().decode('utf-8')

master = '{}:{}'.format(ip, port)

print('Connecting to Scanner database...')
sc = sp.Client(master=master, start_cluster=False, config_path='./config.toml')

print('Running Scanner job...')
example_video_path = 'sample.mp4'
video_stream = sp.NamedVideoStream(sc, 'example', path=example_video_path)

print(db.summarize())

frames = sc.io.Input([video_stream])
r_frame = sc.ops.Resize(frame=frame, width=320, height=240)
output_stream = sp.NamedVideoStream(sc, 'example_frame')
output_op = sc.io.Output(hists, [output_stream])

job_id = sc.run(output_op, sp.PerfParams.estimate())

output_stream.save_mp4('resized_video')

print('Complete!')
