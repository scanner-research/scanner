from __future__ import print_function
import json
import numpy as np
import struct
import cv2
import scipy.io as sio
import scipy.misc
import math


def parse_calibration_data(data):
    return data


def load_calibration_data(path):
    with open(path) as f:
        d = json.load(f)
    return parse_calibration_data(d)


def y_rotation_matrix(theta):
    theta = theta - np.pi / 2
    return np.matrix([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])

def midpoint_transformation(calibration, cam_left_idx, cam_right_idx):
    num_cameras = calibration['numCameras']
    camera_radius = calibration['ringRadius']
    fov = calibration['fov'] * np.pi / 180.0

    def radians_to_pos(rad):
        pos =  np.array([camera_radius * math.cos(rad),
                         0,
                         camera_radius * math.sin(rad)])
        pos.shape = (3, 1)
        return pos

    cam_left_radians = cam_left_idx / (1.0 * num_cameras) * 2 * np.pi
    cam_left_pos = radians_to_pos(cam_left_radians)
    cam_left_rotation = y_rotation_matrix(cam_left_radians)
    cam_left_ext_inv = np.hstack((cam_left_rotation, cam_left_pos))
    cam_left_ext_inv = np.vstack((cam_left_ext_inv, np.array([0, 0, 0, 1])))
    cam_left_ext = np.linalg.inv(cam_left_ext_inv)

    cam_right_radians = cam_right_idx / (1.0 * num_cameras) * 2 * np.pi
    cam_right_pos = radians_to_pos(cam_right_radians)
    cam_right_rotation = y_rotation_matrix(cam_right_radians)
    cam_right_ext_inv = np.hstack((cam_right_rotation, cam_right_pos))
    cam_right_ext_inv = np.vstack((cam_right_ext_inv, np.array([0, 0, 0, 1])))
    cam_right_ext = np.linalg.inv(cam_right_ext_inv)

    cam_mid_radians = cam_left_radians * 0.5 + cam_right_radians * 0.5
    cam_mid_pos = radians_to_pos(cam_mid_radians)
    cam_mid_rotation = y_rotation_matrix(cam_mid_radians)
    cam_mid_ext_inv = np.hstack((cam_mid_rotation, cam_mid_pos))
    cam_mid_ext_inv = np.vstack((cam_mid_ext_inv, np.array([0, 0, 0, 1])))
    cam_mid_ext = np.linalg.inv(cam_mid_ext_inv)

    # transformation to get projection matrix
    image = scipy.misc.imread('cam-1-frame-100.png')
    imageLeft = image
    imageRight = scipy.misc.imread('cam-2-frame-100.png')

    width = image.shape[1]
    height = image.shape[0]
    aspect_ratio = width / (1.0*height)

    cam_space = np.matrix([[1.0/(width/2), 0, -1.0],
                           [0, 1.0/(height/2), -1.0],
                           [0, 0, 1]])
    cam_space_inv = np.matrix([[width/2, 0, 0, width/2],
                               [0, height/2, 0, height/2],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    cam_space = np.vstack((cam_space, np.array([0, 0, 1])))
    ez = np.tan(fov / 2 )
    perspective_inv = np.matrix([[ez * aspect_ratio, 0, 0, 0],
                                 [0, ez, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
    perspective = np.matrix([[1 / (ez * aspect_ratio), 0, 0, 0],
                             [0, 1 / ez, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1, 0]])

    left_to_mid_matrix1 = perspective_inv * cam_space
    left_to_mid_matrix = perspective * cam_mid_ext * cam_left_ext_inv
    right_to_mid_matrix1 = perspective_inv * cam_space
    right_to_mid_matrix = perspective * cam_mid_ext * cam_right_ext_inv

    out = np.zeros((height, width, 3), np.uint8)
    out[:] = (0, 0, 0)
    source_points = np.matrix([[0.0, 0.0],
                               [width, 0],
                               [0, height],
                               [width, height]], dtype=np.float32)
    left_destination_points = np.zeros((4, 2), dtype=np.float32)
    right_destination_points = np.zeros((4, 2), dtype=np.float32)
    infz = 1000000000000.0
    for i in range(4):
        one = np.array([1])
        one.shape = (1, 1)
        p = np.hstack((source_points[i], one))
        zl = left_to_mid_matrix1 * p.T
        zl[3] = 1/infz
        zl *= 1/zl[3]
        zl = left_to_mid_matrix * zl
        zl /= zl[3]
        zl[0] = zl[0] * width / 2 + width / 2
        zl[1] = zl[1] * height / 2 + height / 2

        zr = right_to_mid_matrix1 * p.T
        zr[3] = 1/infz
        zr *= 1/zr[3]
        zr = right_to_mid_matrix * zr
        zr /= zr[3]
        zr[0] = zr[0] * width / 2 + width / 2
        zr[1] = zr[1] * height / 2 + height / 2
        left_destination_points[i,0] = zl[0]
        left_destination_points[i,1] = zl[1]
        right_destination_points[i,0] = zr[0]
        right_destination_points[i,1] = zr[1]

    left_to_mid_warp = cv2.getPerspectiveTransform(source_points,
                                                   left_destination_points)
    right_to_mid_warp = cv2.getPerspectiveTransform(source_points,
                                                    right_destination_points)
    out = cv2.warpPerspective(imageLeft, left_to_mid_warp, (width, height), out)
    cv2.imwrite('warpLeft.jpg', out)
    out = cv2.warpPerspective(imageRight, right_to_mid_warp, (width, height), out)
    cv2.imwrite('warpRight.jpg', out)


if __name__ == "__main__":
    calibration = load_calibration_data('calibration.json')
    midpoint_transformation(calibration, 1, 2)
