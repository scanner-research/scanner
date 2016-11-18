from __future__ import print_function
import numpy as np
import sys
import scipy.misc
from scanner import JobLoadException
from collections import defaultdict
from pprint import pprint
import os
import toml
import scanner
import struct
import math
import json
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


def dataset_list_to_panel_cams(dataset_paths):
    panel_cams = []
    for p in dataset_paths:
        file_name = os.path.splitext(os.path.basename(p))[0]
        print(file_name)
        parts = file_name.split("_")
        panel_idx = int(parts[1])
        camera_idx = int(parts[2])
        panel_cams.append((panel_idx, camera_idx))
    return panel_cams


def node_map_to_relative_node(heat_map):
    heat_map_resized = cv.resize(
        heat_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    x, y = np.unravel_index(heat_map_resized.argmax(),
                            heat_map_resized.shape)
    score = heat_map_resized[x, y]
    return [x, y, score]


def node_maps_to_pose(offset, heat_maps):
    nodes = np.zeros((14, 3))
    for part in range(14):
        node_map = heat_maps[part, :, :]
        nodes[part, :] = node_map_to_relative_node(node_map)
    nodes[:, 0] = nodes[:, 0] - (368.0 / 2) + offset[0]
    nodes[:, 1] = nodes[:, 1] - (368.0 / 2) + offset[1]
    return nodes


def parse_cpm_data(person_centers_job, joint_results_job):
    sampled_frames = defaultdict(list)
    person_centers = defaultdict(list)
    for out in person_centers_job.as_outputs():
        vi = out['video']
        sampled_frames[vi] += out['frames']
        person_centers[vi] += out['buffers']

    i = 0
    person_poses = defaultdict(list)
    for out in joint_results_job.as_outputs():
        vi = out['video']
        for centers in person_centers[vi]:
            if len(centers) + i > len(out['buffers']):
                break
            poses = []
            for p in range(len(centers)):
                node_maps = out['buffers'][i]
                poses.append(node_maps_to_pose(centers[p], node_maps))
                i += 1
            person_poses[vi].append(poses)
    return sampled_frames, person_centers, person_poses


def nest_in_panel_cam(panel_cam_list, data):
    nested = defaultdict(dict)
    for vi, d in data.iteritems():
        vi = int(vi)
        panel_idx, camera_idx = panel_cam_list[vi]
        nested[panel_idx][camera_idx] = d
    return nested


def draw_pose(frame, person):
    part_str = ['head', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
                'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'bkg']
    limbs = np.array(
        [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11, 12, 13, 13, 14])
    num_limbs = len(limbs)/2
    limbs = limbs.reshape((num_limbs, 2)).astype(np.int)
    stickwidth = 6
    colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
              [170, 255, 0], [255, 170, 0], [255, 0, 0], [255, 0, 170],
              [170, 0, 255]]

    for part in range(14):
    #for part in [0]: 
        cv.circle(
            frame, (int(person[part, 1]),
                    int(person[part, 0])),
            3, (0, 0, 0), -1)
    for l in range(limbs.shape[0]):
        cur_frame = frame.copy()
        X = person[limbs[l,:]-1, 0]
        Y = person[limbs[l,:]-1, 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv.ellipse2Poly((int(mY),int(mX)),
                                  (int(length/2), stickwidth),
                                  int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_frame, polygon, colors[l])
        frame = frame * 0.4 + cur_frame * 0.6 # for transparency


def save_drawn_poses_on_frames(video_paths,
                               sampled_frames,
                               person_centers,
                               person_poses):
    for vi in sampled_frames.keys():
        cap = cv.VideoCapture(video_paths[int(vi)])
        s_fi = sampled_frames[vi]
        s_poses = person_poses[vi]
        s_centers = person_centers[vi]
        curr_fi = 0
        print('Generating ' + str(len(s_fi)) + ' frames for video ' + vi)
        for fi, poses, centers in zip(s_fi, s_poses, s_centers):
            if not cap.isOpened():
                break
            while cap.isOpened():
                r, frame = cap.read()
                curr_fi += 1
                if curr_fi - 1 == fi:
                    break
            scale = frame.shape[0] / 368.0
            cs = [[c[0] * scale, c[1] * scale] for c in centers]
            for center in cs:
                cv.circle(
                    frame, (int(center[0]), int(center[1])),
                    5, (0, 255, 255), -1)
            for person in poses:
                # [head, rsho, rwri, lsho, lwri, rank, lank]
                person *= scale
                draw_pose(frame, person)

            if fi % 100 == 0:
                print('At frame ' + str(fi) + '...')
            scipy.misc.toimage(frame[:,:,::-1]).save(
                'imgs/frames{:04d}.jpg'.format(fi))
            if not cap.isOpened():
                break


def parse_calibration_data(data):
    calibration_data = json.loads(data)
    calib = {}
    calib['panels'] = range(1, 21)
    calib['nodes'] = range(1, 25)
    calib['cameras'] = defaultdict(dict)
    for cam in calibration_data['cameras']:
        panel_idx = cam['panel']
        node_idx = cam['node']
        calib['cameras'][panel_idx][node_idx] = cam
    return calib


def write_extrinsic_params(calibration_data,
                           top_level_path):
    for panel_idx in panels:
        for camera_idx in cameras:
            ext_file = os.path.join(
                top_level_path,
                '{:02d}_{:02d}_ext.txt'.format(panel_idx, camera_idx))
            c = calibration_data['cameras'][panel_idx][camera_idx]
            with open(ext_file, 'w') as f:
                def wr(s):
                    f.write(str(s) + ' ')

            # WIP



def write_pose_detections(calibration_data,
                          poses,
                          frame,
                          top_level_path):
    directory = os.path.join(top_level_path, 'poseDetect_pm', 'vga_25')
    output_file_name = os.path.join(
        directory, 'poseDetectMC_{:08d}.txt'.format(frame))

    num_joints = 14
    panels = calibration_data['panels']
    cameras = calibration_data['nodes']
    with open(output_file_name, 'w') as f:
        def wr(s):
            f.write(str(s) + ' ')

        wr('dummy') # dummy, unused
        wr(0.5) # version number
        wr('dummy') # dummy, unused
        wr(480) # processed views, unused
        # For all cameras on all panels
        for panel_idx in panels:
            if not panel_idx in poses:
                continue
            for camera_idx in cameras:
                if not camera_idx in poses[panel_idx]:
                    continue
                wr(frame)
                wr(panel_idx)
                wr(camera_idx)

                people = poses[panel_idx][camera_idx][0]
                num_people = len(people)
                wr(num_people)
                wr(num_joints)
                for person in people:
                    joints = person
                    for j in range(num_joints):
                        wr(joints[j, 0])
                        wr(joints[j, 1])
                        wr(joints[j, 2])


def main():
    if len(sys.argv) != 2:
        print('Usage: cpm.py <dataset_name>')
        exit()

    [dataset_name] = sys.argv[1:]

    data_path = '/bigdata/apoms/panoptic/160422_mafia2'
    calib_path = os.path.join(data_path, 'calibration_160422_mafia2.json')
    with open(calib_path, 'r') as f:
        calib_data = parse_calibration_data(f.read())

    person_centers_job = load_cpm_person_centers(dataset_name, 'person')
    joint_results_job = load_cpm_joint_maps(dataset_name, 'pose')

    sampled_frames, person_centers, person_poses = parse_cpm_data(
        person_centers_job, joint_results_job)

    video_paths = person_centers_job._dataset.video_data.original_video_paths
    panel_cam = dataset_list_to_panel_cams(video_paths)
    nested_poses = nest_in_panel_cam(panel_cam, person_poses)
    write_pose_detections(calib_data, nested_poses, 1000, '.')
    save_drawn_poses_on_frames(
        video_paths, sampled_frames, person_centers, person_poses)

if __name__ == "__main__":
    main()
