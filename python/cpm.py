from __future__ import print_function
import numpy as np
import sys
import scipy.misc
from scanner import JobLoadException
from collections import defaultdict
import os
import toml
import scanner
import struct
import cv2 as cv

db = scanner.Scanner()
import scannerpy.evaluators.types_pb2

@db.loader('frame')
def load_frames(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.uint8))
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
        p = [point.y, point.x]
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
    buf = buf.reshape((15, 46, 46))
    return buf

def node_map_to_relative_node(heat_map):
    heat_map_resized = cv.resize(
        heat_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    return np.unravel_index(heat_map_resized.argmax(), heat_map_resized.shape)

def node_maps_to_pose(offset, heat_maps):
    nodes = np.zeros((14, 2))
    for part in range(14):
        node_map = heat_maps[part, :, :]
        nodes[part, :] = node_map_to_relative_node(node_map)
    nodes[:, 0] = nodes[:, 0] - (368.0 / 2) + offset[0]
    nodes[:, 1] = nodes[:, 1] - (368.0 / 2) + offset[1]
    return nodes

def draw_poses(frame, poses):
    pass

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
    #frames_job = load_frames(dataset_name, scanner.Scanner.base_job_name())
    person_centers_job = load_cpm_person_centers(dataset_name, 'person')
    joint_results_job = load_cpm_joint_maps(dataset_name, 'pose')

    sampled_frames = defaultdict(list)
    person_centers = defaultdict(list)
    for out in person_centers_job.as_outputs():
        vi = out['video']
        sampled_frames[vi] += out['frames']
        person_centers[vi] += out['buffers']
        print(vi, len(sampled_frames[vi]))
        print(vi, len(person_centers[vi]))

    i = 0
    person_poses = defaultdict(list)
    for out in joint_results_job.as_outputs():
        vi = out['video']
        print(vi)
        for centers in person_centers[vi]:
            if len(centers) + i > len(out['buffers']):
                break
            poses = []
            for p in range(len(centers)):
                node_maps = out['buffers'][i]
                poses.append(node_maps_to_pose(centers[p], node_maps))
                i += 1
            person_poses[vi].append(poses)
    print(len(person_poses[vi]))

    frames = defaultdict(list)

    video_names = person_centers_job._dataset.video_data.video_names
    video_paths = person_centers_job._dataset.video_data.original_video_paths

    for vi in sampled_frames.keys():
        cap = cv.VideoCapture(video_paths[int(vi)])
        s_fi = sampled_frames[vi]
        s_poses = person_poses[vi]
        s_centers = person_centers[vi]
        curr_fi = 0
        for fi, poses, centers in zip(s_fi, s_poses, s_centers):
            if not cap.isOpened():
                break
            while cap.isOpened():
                r, frame = cap.read()
                curr_fi += 1
                if curr_fi - 1 == fi:
                    break
            scale = frame.shape[0] / 368.0
            for center in centers:
                cv.circle(
                    frame, (int(center[1] * scale),
                            int(center[0] * scale)),
                    5, (0, 255, 255), -1)
            for person in poses:
                # [head, rsho, rwri, lsho, lwri, rank, lank]
                for part in [0,2,4,5,7,10,13]: 
                #for part in [0]: 
                    cv.circle(
                        frame, (int(person[part, 1] * scale),
                                int(person[part, 0] * scale)),
                        5, (0, 0, 255), -1)

            scipy.misc.toimage(frame[:,:,::-1]).save(
                'imgs/frames{:04d}.jpg'.format(fi))
            if not cap.isOpened():
                break


    # for out in frames_job.as_outputs():
    #     vi = out['video']
    #     s_fi = sampled_frames[vi]
    #     print(out['frames'])
    #     for b in out['buffers']:
    #         #print('frame {}, points: {}'.format(i, b))
    #         # scipy.misc.toimage(b).save(
    #         #scipy.misc.toimage(b[0:3, :, :], cmin=-0.5, cmax=0.5).save(
    #         #scipy.misc.toimage(b[3, :, :], cmin=0.0, cmax=1.0).save(
    #         #scipy.misc.toimage(b[3, :, :]).save(

    #         scipy.misc.toimage(b[0, :, :]).save(
    #             'imgs/centers{:04d}.jpg'.format(i))
    #         i += 1

if __name__ == "__main__":
    main()
