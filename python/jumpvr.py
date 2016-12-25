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
    return np.matrix([[np.cos(theta), 0, -np.sin(theta)],
                      [0, 1, 0],
                      [np.sin(theta), 0, np.cos(theta)]])

def midpoint_transformation(calibration, cam_left_idx, cam_right_idx):
    num_cameras = calibration['numCameras']
    camera_radius = calibration['ringRadius']
    fov = calibration['fov'] * np.pi / 180.0
    aspect_ratio = 1920.0 / 1080.0

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

    cam_mid_radians = (cam_left_radians + cam_right_radians) / 2
    cam_mid_pos = radians_to_pos(cam_mid_radians)
    cam_mid_rotation = y_rotation_matrix(cam_mid_radians)
    cam_mid_ext_inv = np.hstack((cam_mid_rotation, cam_mid_pos))
    cam_mid_ext_inv = np.vstack((cam_mid_ext_inv, np.array([0, 0, 0, 1])))
    cam_mid_ext = np.linalg.inv(cam_mid_ext_inv)
    print(cam_left_radians, cam_right_radians, cam_mid_radians)

    print()
    print(cam_left_ext_inv)
    print(cam_left_ext)
    print(cam_mid_ext)
    # need to compute inverse of camera matrix and then apply mid point camera

    # transformation to get projection matrix
    image = np.zeros((1080, 1920, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    image[:] = (255, 255, 255)

    cam_space = np.matrix([[1.0/(image.shape[0]/2), 0, -1.0],
                           [0, 1.0/(image.shape[1]/2), -1.0],
                           [0, 0, 1]])
    cam_space_inv = np.linalg.inv(cam_space)
    cam_space = np.vstack((cam_space, np.array([0, 0, 1])))
    print(cam_space)
    cam_space_inv = np.vstack((cam_space_inv, np.array([0, 0, 0])))
    cam_space_inv = np.hstack((cam_space_inv, np.reshape(np.array([0, 0, 0, 1]), (4, 1))))
    print(cam_space_inv)
    ez = np.tan(fov / 2 )
    perspective_inv = np.matrix([[ez, 0, 0, 0],
                                 [0, ez, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
    perspective = np.matrix([[1, 0, -1 / ez, 0],
                             [0, 1, -1 / ez, 0],
                             [0, 0, 1, 0],
                             [0, 0, 1 / ez, 0]])
    #print('persp', np.linalg.inv(perspective))

    left_to_mid_matrix = cam_mid_ext * cam_left_ext_inv * perspective_inv * cam_space
    right_to_mid_matrix = cam_mid_ext * cam_right_ext_inv * perspective_inv * cam_space
    print(left_to_mid_matrix)
    print(right_to_mid_matrix)

    # Create a blank 300x300 black image

    out = np.zeros((1080, 1920, 3), np.uint8)
    out[:] = (0, 0, 0)
    source_points = np.matrix([[0.0, 0.0],
                               [image.shape[1], 0],
                               [0, image.shape[0]],
                               [image.shape[1], image.shape[0]]], dtype=np.float32)
    left_destination_points = np.zeros((4, 2), dtype=np.float32)
    right_destination_points = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        one = np.array([1])
        one.shape = (1, 1)
        p = np.hstack((source_points[i], one))
        print(p)
        zl = left_to_mid_matrix * p.T
        print('zl', zl)
        zl /= zl[3]
        zl = perspective * zl
        zl /= zl[3]
        zl[0] = zl[0] * image.shape[0] + image.shape[0]
        zl[1] = zl[1] * image.shape[1] + image.shape[1]
        zr = right_to_mid_matrix * p.T
        zr /= zr[3]
        zr[0] = zr[0] * image.shape[0] + image.shape[0]
        zr[1] = zr[1] * image.shape[1] + image.shape[1]
        print('left', p, 'unpersp', perspective_inv * cam_space * p.T, 'center', zl)
        left_destination_points[i,0] = zl[0]
        left_destination_points[i,1] = zl[1]
        right_destination_points[i,0] = zr[0]
        right_destination_points[i,1] = zr[1]

    print(left_destination_points)
    # destination_points = np.matrix([[], [], [], []])
    left_to_mid_warp = cv2.getPerspectiveTransform(source_points,
                                                   left_destination_points)
    right_to_mid_warp = cv2.getPerspectiveTransform(source_points,
                                                    right_destination_points)
    right_to_mid_matrix = np.identity(3)
    right_to_mid_matrix[0, 2] = 505
    right_to_mid_matrix[1, 2] = 250 
    right_to_mid_matrix[2, 0] = 0.0001
    right_to_mid_matrix[2, 1] = 0.0001
    #print(right_to_mid_matrix)
    out = cv2.warpPerspective(image, left_to_mid_warp, (1920, 1080), out)
    #print(image)
    #print(out)
    cv2.imwrite('warp.jpg', out)


if __name__ == "__main__":
    calibration = load_calibration_data('calibration.json')
    midpoint_transformation(calibration, 10, 11)
