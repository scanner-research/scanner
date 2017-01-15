from __future__ import print_function
from pprint import pprint
from collections import defaultdict
from numpy import linalg as LA
import pickle
import numpy as np
import struct
import sys
import scanner
import cv2
import os.path
import scipy.misc
import extract_frames_scanner

db = scanner.Scanner()
from scannerpy import metadata_pb2
from scannerpy.evaluators import types_pb2
import scannerpy.evaluators.types_pb2 as types

@db.loader('joint_centers')
def load_cpm2_joint_centers(buf, metadata):
    if len(buf) == 0: return []
    (num_bodies,) = struct.unpack("=Q", buf[:8])
    buf = buf[8:]
    bodies = []
    for i in range(num_bodies):
        (num_joints,) = struct.unpack("=Q", buf[:8])
        assert(num_joints == 15)
        buf = buf[8:]
        joints = np.zeros((15, 3))
        for i in range(num_joints):
            point_size, = struct.unpack("=i", buf[:4])
            buf = buf[4:]
            point = types.Point()
            point.ParseFromString(buf[:point_size])
            buf = buf[point_size:]
            joints[i, 0] = point.y
            joints[i, 1] = point.x
            joints[i, 2] = point.score
        bodies.append(joints)
    return bodies


def parse_cpm2_data(joint_results_job, scale):
    sampled_frames = defaultdict(list)
    person_poses = defaultdict(list)
    for out in joint_results_job.as_outputs():
        vi = out['table']
        sampled_frames[vi] = out['frames']
        person_poses[vi] += out['buffers']
    by_frame = defaultdict(lambda: defaultdict(list))
    video_ids = []
    for vi in sampled_frames.keys():
        video_ids.append(vi)
        zipped = zip(sampled_frames[vi], person_poses[vi])
        for frame, poses in zipped:
            centers = [(0, 0) for i in range(len(poses))]
            people = []
            for c, p in zip(centers, poses):
                people.append({
                    'center': c,
                    'pose': p,
                })
            by_frame[vi][frame] += people
    return video_ids, by_frame


def iou(bl, br):
    blx1 = bl[0]
    bly1 = bl[1]
    blx2 = bl[2]
    bly2 = bl[3]

    brx1 = br[0]
    bry1 = br[1]
    brx2 = br[2]
    bry2 = br[3]

    x1 = max(blx1, brx1)
    y1 = max(bly1, bry1)
    x2 = min(blx2, brx2)
    y2 = min(bly2, bry2)

    if (x1 >= x2 or y1 >= y2):
        return 0

    intersection = (y2 - y1) * (x2 - x1)
    _union = ((blx2 - blx1) * (bly2 - bly1) +
              (brx2 - brx1) * (bry2 - bry1) -
              intersection)
    if _union == 0:
        return 0
    return intersection / _union


def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    elif len(boxes) == 1:
        return boxes

    print('nms',boxes)
    npboxes = np.array(boxes[0])
    for box in boxes[1:]:
        npboxes = np.vstack((npboxes, box))
    boxes = npboxes
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(
            ([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def visualize_frames(dataset_name, video, v_frames, nms_bboxes, output_dir):
    # Perform nms on boxes
    bboxes = nms_bboxes[video]
    print(bboxes)
    # Extract frames
    frames = [(video, v_frames)]
    extract_frames_scanner.get_frames(dataset_name, frames,
                                      '/tmp/scanner_frames')
    image_template = '/tmp/scanner_frames/{}_{:07d}.jpg'
    for i, frame in enumerate(v_frames):
        image = cv2.imread(image_template.format(video, frame))
        #r, image = cap.read()
        boxes = bboxes[i]
        print('Frame', frame, 'boxes', boxes)
        for bbox in boxes:
            bbox = np.array(bbox).astype(int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 0, 255), 3)
        file_name = video + "_frame_" + str(frame) + ".jpg"
        file_path = os.path.join(output_dir, file_name)
        scipy.misc.toimage(image[:,:,::-1]).save(file_path)


def main():
    dataset_name = "pier"
    job_name_prefix = "cpm2_test"
    output_column = "base_bboxes"
    job_name_template = job_name_prefix + "_{:d}"
    output_path = 'cpm2_output'
    start_frame = 1500
    end_frame = 1800
    stride = 1

    #scales = [pow(2, x) for x in range(-3, 1, 1)]
    base_scale = 0.51
    scales = np.array([0.25, 0.5, 0.75, 1.0]) * base_scale
    #scales = np.array([0.25, 0.5]) * base_scale
    print(scales)

    opts = {
        'force': True,
        'io_item_size': 32,
        'work_item_size': 16,
        'pus_per_node': 1,
        'env': {}
    }
    for i, scale in enumerate(scales):
        print('Running with scale', scale)
        print('Size {:.1f} x {:.1f}'.format(1280 * scale, 720 * scale))
        job_name = job_name_template.format(i)
        opts['env']['SC_SCALE'] = str(scale)
        opts['env']['SC_START_FRAME'] = str(start_frame)
        opts['env']['SC_END_FRAME'] = str(end_frame)
        opts['env']['SC_STRIDE'] = str(stride)
        rc, t = db.run(dataset_name, 'cpm2', job_name, opts)
        assert(rc == True)
        print('Time', t)

    #write('Loading CPM... ')
    all_bboxes = defaultdict(list)
    for i, scale in enumerate(scales):
        job_name = job_name_template.format(i)
        cpm2_data = load_cpm2_joint_centers(dataset_name, job_name)
        print(cpm2_data)
        _, poses_by_video = parse_cpm2_data(cpm2_data, scale)
        print(poses_by_video)
        for vid, poses_by_frame in poses_by_video.iteritems():
            scale_bboxes = []
            frames = poses_by_frame.keys()
            frames.sort()
            for frame in frames:
                poses = poses_by_frame[frame]
                frame_bboxes = []
                print('Frame', frame, 'poses', len(poses))
                for pose in poses:
                    pose = pose['pose']
                    head = pose[0:2, 0:2]
                    axis = head[1,:] - head[0,:]
                    cross = [-axis[1], axis[0]]
                    if np.abs(LA.norm(cross)) < 0.01 : continue
                    cross /= LA.norm(cross)
                    midpoint = [(head[0,0]+head[1,0])/2.0,
                                (head[0,1]+head[1,1])/2.0]
                    halfnorm = LA.norm(axis) / 2.0
                    pts = [head[0,:], head[1,:]]
                    pts.append(midpoint + cross * halfnorm)
                    pts.append(midpoint - cross * halfnorm)
                    pts = np.array(pts)
                    bbox = np.array([np.min(pts[:,1]), np.min(pts[:,0]),
                                     np.max(pts[:,1]), np.max(pts[:,0]), 2])
                    frame_bboxes.append(bbox)
                scale_bboxes.append(frame_bboxes)
            all_bboxes[vid].append(scale_bboxes)

    # Collect boxes from different runs for same frame into one list
    #bboxes = all_bboxes[0]
    nms_bboxes = defaultdict(list)
    print('keys', all_bboxes.keys())
    print('boxes', nms_bboxes)
    for vi, boxes in all_bboxes.iteritems():
        print(vi)
        print(len(boxes))
        print(len(boxes[0]))
        frames = len(boxes[0])
        runs = len(boxes)
        new_boxes = []
        for fi in range(frames):
            frame_boxes = []
            for r in range(runs):
                frame_boxes += list(boxes[r][fi])
            frame_boxes = nms(frame_boxes, 0.1)
            new_boxes.append(frame_boxes)
        nms_bboxes[vi] = new_boxes

    visualize_frames(dataset_name, '0',
                     range(start_frame, end_frame, stride),
                     nms_bboxes, output_path)



if __name__ == "__main__":
    main()
