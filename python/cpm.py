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
import errno
import json
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'


db = scanner.Scanner()
import scannerpy.evaluators.types_pb2


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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
        p = np.array([point.y, point.x])
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
    y, x = np.unravel_index(heat_map_resized.argmax(),
                            heat_map_resized.shape)
    score = heat_map_resized[y, x]
    return [y, x, score]


def node_maps_to_pose(offset, heat_maps):
    nodes = np.zeros((14, 3))
    for part in range(14):
        node_map = heat_maps[part, :, :]
        nodes[part, :] = node_map_to_relative_node(node_map)
    nodes[:, 0] = nodes[:, 0] - (368.0 / 2) + offset[0]
    nodes[:, 1] = nodes[:, 1] - (368.0 / 2) + offset[1]
    return nodes


def parse_cpm_data(person_centers_job, joint_results_job, scale):
    sampled_frames = defaultdict(list)
    person_centers = defaultdict(list)
    for out in person_centers_job.as_outputs():
        vi = out['video']
        sampled_frames[vi] += out['frames']
        person_centers[vi] += out['buffers']

    person_poses = defaultdict(list)
    for out in joint_results_job.as_outputs():
        vi = out['video']
        i = 0
        for centers in person_centers[vi]:
            poses = []
            if len(centers) + i > len(out['buffers']):
                i += len(centers)
                person_poses[vi].append(poses)
                continue
            for p in range(len(centers)):
                node_maps = out['buffers'][i]
                poses.append(node_maps_to_pose(centers[p], node_maps) * scale)
                centers[p] *= scale
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
        cv.circle(
            frame, (int(person[part, 1]),
                    int(person[part, 0])),
            3, (0, 0, 0), -1)
    for l in range(limbs.shape[0]):
        cur_frame = frame.copy()
        X = person[limbs[l,:]-1, 1]
        Y = person[limbs[l,:]-1, 0]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
        polygon = cv.ellipse2Poly((int(mX),int(mY)),
                                  (int(length/2), stickwidth),
                                  int(angle), 0, 360, 1)
        cv.fillConvexPoly(cur_frame, polygon, colors[l])
        frame = frame * 0.4 + cur_frame * 0.6 # for transparency
    return frame


def save_drawn_poses_on_frames(video_paths,
                               video_index_to_panel_cam,
                               sampled_frames,
                               person_centers,
                               person_poses):
    for vi in sampled_frames.keys():
        cap = cv.VideoCapture(video_paths[int(vi)])
        s_fi = sampled_frames[vi]
        s_poses = person_poses[vi]
        s_centers = person_centers[vi]
        curr_fi = 0
        panel, camera = video_index_to_panel_cam[int(vi)]
        print('Generating ' + str(len(s_fi)) + ' frames for video ' + vi +
              ', panel ' + str(panel) + ', camera ' + str(camera))
        for fi, poses, centers in zip(s_fi, s_poses, s_centers):
            if not cap.isOpened():
                break
            while cap.isOpened():
                r, frame = cap.read()
                curr_fi += 1
                if curr_fi - 1 == fi:
                    break
            cs = centers
            for center in cs:
                cv.circle(
                    frame, (int(center[1]), int(center[0])),
                    5, (0, 255, 255), -1)
            for person in poses:
                # [head, rsho, rwri, lsho, lwri, rank, lank]
                frame = draw_pose(frame, person)

            if fi % 100 == 0:
                print('At frame ' + str(fi) + '...')
            scipy.misc.toimage(frame[:,:,::-1]).save(
                'imgs/{:02d}_{:02d}_frame_{:04d}.jpg'.format(panel, camera, fi))


def parse_calibration_data(calibration_data):
    calib = {}
    calib['panels'] = range(1, 21)
    calib['nodes'] = range(1, 25)
    calib['cameras'] = defaultdict(dict)
    for cam in calibration_data['cameras']:
        panel_idx = cam['panel']
        node_idx = cam['node']
        calib['cameras'][panel_idx][node_idx] = cam
    return calib


# Taken from https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
def rotation2quaternion(M):
    ''' Calculate quaternion corresponding to given rotation matrix

    Parameters
    ----------
    M : array-like
      3x3 rotation matrix

    Returns
    -------
    q : (4,) array
      closest quaternion to input matrix, having positive q[0]

    Notes
    -----
    Method claimed to be robust to numerical errors in M

    Constructs quaternion by calculating maximum eigenvector for matrix
    K (constructed from input `M`).  Although this is not tested, a
    maximum eigenvalue of 1 corresponds to a valid rotation.

    A quaternion q*-1 corresponds to the same rotation as q; thus the
    sign of the reconstructed quaternion is arbitrary, and we return
    quaternions with positive w (q[0]).

    References
    ----------
    * http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    * Bar-Itzhack, Itzhack Y. (2000), "New method for extracting the
      quaternion from a rotation matrix", AIAA Journal of Guidance,
      Control and Dynamics 23(6):1085-1087 (Engineering Note), ISSN
      0731-5090

    Examples
    --------
    >>> import numpy as np
    >>> q = mat2quat(np.eye(3)) # Identity rotation
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = mat2quat(np.diag([1, -1, -1]))
    >>> np.allclose(q, [0, 1, 0, 0]) # 180 degree rotn around axis 0
    True

    '''
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q


### From panutils.py
def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].
    
    Roughly, x = K*(R*X + t) + distortion
    
    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """
    
    x = np.asarray(R*X + t)
    
    x[0:2,:] = x[0:2,:]/x[2,:]
    
    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]
    
    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]
    
    return x


def get_uniform_camera_order():
    """ Returns uniformly sampled camera order as a list of tuples [(panel,node), (panel,node), ...]."""
    panel_order =[1,19,14,6,16,9,5,10,18,15,3,8,4,20,11,13,7,2,17,12,9,5,6,3,15,2,12,14,16,10,4,13,20,8,17,19,18,9,4,6,1,20,1,11,7,7,14,15,3,2,16,13,3,15,17,9,20,19,8,11,5,8,18,10,12,19,5,6,16,12,4,6,20,13,4,10,15,12,17,17,16,1,5,3,2,18,13,16,8,19,13,11,10,7,3,2,18,10,1,17,10,15,14,4,7,9,11,7,20,14,1,12,1,6,11,18,7,8,9,3,15,19,4,16,18,1,11,8,4,10,20,13,6,16,7,6,16,17,12,5,17,4,8,20,12,17,14,2,19,14,18,15,11,11,9,9,2,13,5,15,20,18,8,3,19,11,9,2,13,14,5,9,17,9,7,6,12,16,18,17,13,15,17,20,4,2,2,12,4,1,16,4,11,1,16,12,18,9,7,20,1,10,10,19,5,8,14,8,4,2,9,20,14,17,11,3,12,3,13,6,5,16,3,5,10,19,1,11,13,17,18,2,5,14,19,15,8,8,9,3,6,16,15,18,20,4,13,2,11,20,7,13,15,18,10,20,7,5,2,15,6,13,4,17,7,3,19,19,3,10,2,12,10,7,7,12,11,19,8,9,6,10,6,15,10,11,3,16,1,5,14,6,5,13,20,14,4,18,10,14,14,1,19,8,14,19,3,6,6,3,13,17,8,20,15,18,2,2,16,5,19,15,9,12,19,17,8,9,3,7,1,12,7,13,1,14,5,12,11,2,16,1,18,4,18,10,16,11,7,5,1,16,9,4,15,1,7,10,14,3,2,17,13,19,20,15,10,4,8,16,14,5,6,20,12,5,18,7,1,8,11,5,13,1,16,14,18,12,15,2,12,3,8,12,17,8,20,9,2,6,9,6,12,3,20,15,20,13,3,14,1,4,8,6,10,7,17,13,18,19,10,20,12,19,2,15,10,8,19,11,19,11,2,4,6,2,11,8,7,18,14,4,12,14,7,9,7,11,18,16,16,17,16,15,4,15,9,17,13,3,6,17,17,20,19,11,5,3,1,18,4,10,5,9,13,1,5,9,6,14]
    node_order = [1,14,3,15,12,12,8,6,13,12,12,17,7,17,21,17,4,6,12,18,2,18,5,4,2,17,12,10,18,8,18,5,10,10,17,1,18,7,12,9,13,5,6,18,16,9,16,8,8,10,21,22,16,16,21,16,14,6,14,11,11,20,4,22,4,22,20,19,15,15,15,12,2,2,3,3,20,22,5,9,3,16,23,22,20,8,8,9,2,16,14,16,16,14,1,13,16,12,10,15,18,6,13,10,7,10,4,1,7,21,8,6,4,7,9,10,11,8,4,6,10,4,5,6,21,21,6,6,19,20,20,20,14,19,22,22,23,19,9,15,23,23,23,23,19,2,8,2,8,19,19,23,23,19,19,23,24,24,2,14,12,2,12,14,12,2,14,15,11,6,6,21,4,5,5,4,2,10,5,10,7,3,7,9,8,9,3,7,9,9,7,2,5,5,5,5,7,8,8,4,7,11,9,7,5,3,5,7,6,8,9,8,7,8,8,3,8,7,6,11,7,2,9,9,2,11,12,7,4,6,6,7,4,4,9,18,1,5,6,5,10,11,5,9,6,11,12,1,10,11,6,9,7,11,5,1,2,12,11,11,3,3,21,11,10,2,3,10,11,19,5,11,13,12,20,13,3,5,9,11,8,4,6,4,7,12,10,8,11,19,14,23,10,1,3,12,4,3,10,9,2,3,20,4,11,2,20,20,2,23,10,3,22,22,1,12,12,21,4,22,23,22,18,10,18,22,11,3,18,13,18,3,3,13,2,1,3,20,20,4,20,14,14,20,20,14,14,22,18,21,20,22,20,22,9,22,21,21,22,21,22,20,21,21,21,21,23,17,21,13,20,13,13,15,17,1,23,23,23,18,13,16,15,19,17,17,22,21,17,14,1,13,13,14,14,16,19,17,18,1,13,18,24,19,16,13,18,18,15,23,17,14,19,17,1,19,13,19,1,15,17,13,23,13,19,24,15,15,19,15,17,1,16,24,21,23,14,24,15,24,24,1,16,15,24,1,17,17,15,24,1,16,16,19,13,15,22,24,23,17,16,18,1,24,24,24,17,24,24,17,16,24,14,15,16,15,24,24,24,18]
    
    return zip(panel_order, node_order)
### 


def write_extrinsic_params(calibration_data,
                           top_level_path):
    panels = calibration_data['panels']
    cameras = calibration_data['nodes']
    for panel_idx in panels:
        for camera_idx in cameras:
            if not ((panel_idx in calibration_data['cameras']) and
                    (camera_idx in calibration_data['cameras'][panel_idx])):
                continue
            ext_file = os.path.join(
                top_level_path,
                '{:02d}_{:02d}_ext.txt'.format(panel_idx, camera_idx))
            c = calibration_data['cameras'][panel_idx][camera_idx]
            with open(ext_file, 'w') as f:
                def wr(s):
                    f.write(str(s) + ' ')

                quat = rotation2quaternion(np.array(c['R']))
                wr(quat[0])
                wr(quat[1])
                wr(quat[2])
                wr(quat[3])

                center = c['t']
                wr(center[0][0])
                wr(center[1][0])
                wr(center[2][0])


def write_pose_detections(calibration_data,
                          poses,
                          frame,
                          top_level_path):
    directory = os.path.join(top_level_path, 'poseDetect_pm_org', 'vga_25')
    mkdir_p(directory)
    output_file_name = os.path.join(
        directory, 'poseDetectMC_{:08d}.txt'.format(frame))

    num_joints = 14
    panels = calibration_data['panels']
    cameras = calibration_data['nodes']
    with open(output_file_name, 'w') as f:
        def wr(s):
            f.write(str(s) + ' ')

        wr('ver') # dummy, unused
        wr(0.5) # version number
        f.write('\n')
        wr('processedViews') # dummy, unused
        wr(480) # processed views, unused
        f.write('\n')
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
                f.write('\n')

                people = poses[panel_idx][camera_idx][0]
                num_people = len(people)
                wr(num_people)
                wr(num_joints)
                for person in people:
                    joints = person
                    for j in range(num_joints):
                        wr(joints[j, 1])
                        wr(joints[j, 0])
                        wr(joints[j, 2])
                    f.write('\n')

def draw_3d_poses(calibration_data, data_path, output_directory, dataset_name,
                  frame_number):
    calib = calibration_data
    seq_name = dataset_name

    vga_skel_json_path = os.path.join(output_directory,
                                      'body3DPSRecon_json',
                                      str(frame_number))
    vga_img_path = os.path.join(data_path, 'vgaImgs')

    hd_skel_json_path = os.path.join(output_directory, 'hdPose3d_stage1')
    hd_img_path = os.path.join(data_path, 'hdImgs')

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k,cam in cameras.iteritems():
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))

    # Select the first 10 VGA cameras in a uniformly sampled order
    #cams = get_uniform_camera_order()[0:10]
    cams = [(1, 1), (1, 4), (1, 9), (1, 18)]
    sel_cameras = [cameras[cam].copy() for cam in cams]

    # Edges between joints in the skeleton
    edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],
                      [13,14],[14,15],[1,10],[10,11],[11,12]])-1
    colors = plt.cm.hsv(np.linspace(0, 1, 33)).tolist()

    # Frame
    idx = frame_number
    plt.figure(figsize=(15,15))
    for icam in xrange(len(sel_cameras)):
        # Select a camera
        cam = sel_cameras[icam]

        # Load the corresponding frame
        image_path = os.path.join(
            vga_img_path,
            '{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'],
                                                                 cam['node'],
                                                                 idx))
        im = plt.imread(image_path)

        plt.subplot(3,2,icam+1)
        plt.imshow(im)
        currentAxis = plt.gca()
        currentAxis.set_autoscale_on(False)

        try:
            # Load the json file with this frame's skeletons
            skel_json_fname = os.path.join(
                vga_skel_json_path, 'body3DScene_{0:08d}.json'.format(idx))
            with open(skel_json_fname) as dfile:
                bframe = json.load(dfile)

            # Cycle through all detected bodies
            num_bodies = min(len(bframe['bodies']), 40)
            for body in bframe['bodies'][:num_bodies]:
                # There are 15 3D joints, stored as an array
                # [x1,y1,z1,c1,x2,y2,z2,c2,...]
                # where c1 ... c15 are per-joint detection confidences
                skel = np.array(body['joints15']).reshape((-1,4)).transpose()

                # Project skeleton into view (this is like cv2.projectPoints)
                pt = projectPoints(skel[0:3,:],
                                   cam['K'], cam['R'], cam['t'],
                                   cam['distCoef'])

                # Show only points detected with confidence
                valid = skel[3,:]>0.1

                plt.plot(pt[0,valid], pt[1,valid], '.',
                         color=colors[body['id']])

                # Plot edges for each bone
                for edge in edges:
                    if valid[edge[0]] or valid[edge[1]]:
                        plt.plot(pt[0,edge], pt[1,edge],
                                 color=colors[body['id']])

                # Show the joint numbers
                for ip in xrange(pt.shape[1]):
                    if (pt[0,ip]>=0 and
                        pt[0,ip]<im.shape[1] and
                        pt[1,ip]>=0 and
                        pt[1,ip]<im.shape[0]):
                        plt.text(pt[0,ip], pt[1,ip]-5,
                                 '{0}'.format(ip),color=colors[body['id']])

        except IOError as e:
            print('Error reading {0}\n'.format(skel_json_fname)+e.strerror)

        # Also plot selected cameras with (panel,node) label
        for ca in sel_cameras:
            cc = (-ca['R'].transpose()*ca['t'])
            pt = projectPoints(cc,
                               cam['K'], cam['R'], cam['t'],
                               cam['distCoef'])
            if (pt[0]>=0 and
                pt[0]<im.shape[1] and
                pt[1]>=0 and
                pt[1]<im.shape[0]):
                plt.plot(pt[0], pt[1], '.', color=[0,1,0], markersize=5)
                plt.text(pt[0], pt[1],
                         'cam({0},{1})'.format(ca['panel'],ca['node']),
                         color=[1,1,1])

    plt.tight_layout()
    plt.savefig(dataset_name + '_' + str(frame_number) + '.png',
                bbox_inches='tight')


def main():
    if len(sys.argv) != 2:
        print('Usage: cpm.py <dataset_name>')
        exit()

    [dataset_name] = sys.argv[1:]

    frame_number = 1000
    output_path = 'cpm_output_' + dataset_name
    mkdir_p(output_path)
    data_path = '/bigdata/apoms/panoptic/' + dataset_name
    calib_path = os.path.join(data_path,
                              'calibration_{:s}.json'.format(dataset_name))
    with open(calib_path, 'r') as f:
        raw_calib_data = json.loads(f.read())
        calib_data = parse_calibration_data(raw_calib_data)

    #person_centers_job = load_cpm_person_centers(dataset_name, 'person')
    #joint_results_job = load_cpm_joint_maps(dataset_name, 'pose')

    scale = 480 / 368.0
    #sampled_frames, person_centers, person_poses = parse_cpm_data(
    #    person_centers_job, joint_results_job, scale)

    #video_paths = person_centers_job._dataset.video_data.original_video_paths
    #panel_cam_mapping = dataset_list_to_panel_cams(video_paths)
    #nested_poses = nest_in_panel_cam(panel_cam_mapping, person_poses)
    #write_extrinsic_params(calib_data, output_path)
    #write_pose_detections(calib_data, nested_poses, 1000, output_path)
    # save_drawn_poses_on_frames(
    #     video_paths, panel_cam_mapping,
    #     sampled_frames, person_centers, person_poses)

    draw_3d_poses(raw_calib_data, data_path, output_path, dataset_name,
                  frame_number)


if __name__ == "__main__":
    main()
