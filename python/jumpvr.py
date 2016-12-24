from __future__ import print_function
import json
import numpy as np
import struct
import cv2
import scipy.io as sio
import math


def parse_calibration_data(data):
    return data


def load_calibration_data(path):
    with open(path) as f:
        d = json.load(f)
    return parse_calibration_data(d)


def y_rotation_matrix(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])

def midpoint_transformation(calibration, cam_left_idx, cam_right_idx):
    camera_radius = calibration.numCameras
    camera_radius = calibration.cameraRadius

    def radians_to_pos(rad):
        return np.array([camera_radius * math.cos(rad),
                         0,
                         camera_radius * math.sin(rad)])

    cam_left_radians = cam_left_idx / (1.0 * num_cameras) * 2 * math.PI
    cam_left_pos = radians_to_pos(cam_left_radians)
    cam_left_rotation = y_rotation_matrix(cam_left_radians)
    cam_left_ext_inv = np.hstack((cam_left_rotation, cam_left_pos.T))
    cam_left_ext = np.linalg.inv(np.vstack((cam_left_ext_inv,
                                            np.array([0, 0, 0, 1]))))
    cam_left_ext = cam_left_ext[0:3]

    cam_right_radians = cam_right_idx / (1.0 * num_cameras) * 2 * math.PI
    cam_right_pos = radians_to_pos(cam_right_radians)
    cam_right_rotation = y_rotation_matrix(cam_right_radians)
    cam_right_ext_inv = np.hstack(cam_right_rotation, cam_right_pos.T)
    cam_right_ext = np.linalg.inv(np.vstack((cam_right_ext_inv,
                                             np.array([0, 0, 0, 1]))))
    cam_right_ext = cam_right_ext[0:3]

    cam_mid_radians = (cam_left_radians + cam_right_radians) / 2
    cam_mid_pos = radians_to_pos(cam_mid_radians)
    cam_mid_rotation = y_rotation_matrix(cam_mid_radians)
    cam_mid_ext_inv = np.hstack(cam_mid_rotation, cam_mid_pos.T)
    cam_mid_ext = np.linalg.inv(np.vstack((cam_mid_ext_inv,
                                           np.array([0, 0, 0, 1]))))
    cam_mid_ext = cam_mid_ext[0:3]
    print(cam_left_radians, cam_right_radians, cam_mid_radians)

    # need to compute inverse of camera matrix and then apply mid point camera
    # transformation to get projection matrix


if __name__ == "__main__":
    calibration = load_calibration_data('calibration.json')
    midpoint_transformation(calibration, 10, 11)
