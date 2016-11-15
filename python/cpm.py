from __future__ import print_function
import numpy as np
import sys
import scipy.misc
from scanner import JobLoadException
import os
import toml
import scanner
import struct

db = scanner.Scanner()
import scannerpy.evaluators.types_pb2

@db.loader('frame')
def load_frames(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.uint8))
    print(buf.shape)
    buf = np.squeeze(buf.reshape((480, 640, 3)))
    np.transpose(buf, (2, 0, 1))
    return buf

@db.loader('net_input')
def load_cpm_person_net_input(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    print(buf.shape)
    buf = np.squeeze(buf.reshape((3, 368, -1)))
    return buf

@db.loader('Mconv7_stage4')
def load_cpm_person_heat_map(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    print(buf.shape)
    buf = np.squeeze(buf.reshape((1, 1, 46, -1)))
    return buf

@db.loader('centers')
def load_cpm_person_centers(buf, metadata):
    (num_points,) = struct.unpack("=Q", buf[:8])
    buf = buf[8:]
    points = []
    for i in range(num_points):
        point_size, = struct.unpack("=i", buf[:4])
        buf = buf[4:]
        point = scannerpy.evaluators.types_pb2.Point()
        point.ParseFromString(buf[:point_size])
        buf = buf[point_size:]
        p = [point.x, point.y]
        points.append(p)
    return points

@db.loader('cpm_input')
def load_cpm_input(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    buf = np.squeeze(buf.reshape((4, 368, 368)))
    return buf

@db.loader('joint_maps')
def load_cpm_joint_maps(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    print(buf.shape)
    buf = buf.reshape((15, 46, 46))
    print(buf)
    return buf

def main():
    if len(sys.argv) != 3:
        print('Usage: cpm.py <dataset_name> <job_name>')
        exit()

    [dataset_name, job_name] = sys.argv[1:]
    #result = load_cpm_person_heat_map(dataset_name, job_name)
    #result = load_frames(dataset_name, job_name)
    #result = load_cpm_person_net_input(dataset_name, job_name)
    #result = load_cpm_person_centers(dataset_name, job_name)
    #result = load_cpm_input(dataset_name, job_name)
    result = load_cpm_joint_maps(dataset_name, job_name)

    i = 0
    for out in result.as_outputs():
        for b in out['buffers']:
            #print('frame {}, points: {}'.format(i, b))
            # scipy.misc.toimage(b).save(
            #scipy.misc.toimage(b[0:3, :, :], cmin=-0.5, cmax=0.5).save(
            #scipy.misc.toimage(b[3, :, :], cmin=0.0, cmax=1.0).save(
            #scipy.misc.toimage(b[3, :, :]).save(
            scipy.misc.toimage(b[0, :, :]).save(
                'imgs/centers{:04d}.jpg'.format(i))
            i += 1

if __name__ == "__main__":
    main()
