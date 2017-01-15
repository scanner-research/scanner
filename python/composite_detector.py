import toml
import scanner
import struct
import numpy as np
from collections import defaultdict
from numpy import linalg as LA
import extract_frames_scanner
import cv2
import os.path
import scipy.misc
from timeit import default_timer as now
import sys

db = scanner.Scanner()
import scannerpy.evaluators.types_pb2 as types

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle',
           'airplane', 'bus','train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
           'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase',
           'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier','toothbrush')
PERSON = CLASSES.index('person')


def write(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def write_timer(start):
    write('{:.3f}\n'.format(now() - start))


@db.loader('bboxes')
def load_bboxes(buf, metadata):
    try:
        (num_bboxes,) = struct.unpack("=Q", buf[:8])
        buf = buf[8:]
        bboxes = []
        for i in range(num_bboxes):
            bbox_size, = struct.unpack("=i", buf[:4])
            buf = buf[4:]
            box = types.BoundingBox()
            box.ParseFromString(buf[:bbox_size])
            bboxes.append(box)
            buf = buf[bbox_size:]
        return bboxes
    except struct.error:
        return []


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


def parse_cpm2_data(joint_results_job):
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


def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    elif len(boxes) == 1:
        return boxes

    #print('nms',boxes)
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
    # print(bboxes)
    # Extract frames
    frames = [(video, v_frames)]
    extract_frames_scanner.get_frames(dataset_name, frames,
                                      '/tmp/scanner_frames')
    image_template = '/tmp/scanner_frames/{}_{:07d}.jpg'
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0)
        ]
    for i, frame in enumerate(v_frames):
        image = cv2.imread(image_template.format(video, frame))
        #r, image = cap.read()
        boxes = bboxes[i]
        # print('Frame', frame, 'boxes', boxes)
        for bbox in boxes:
            bbox = np.array(bbox).astype(int)

            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          colors[bbox[4]], 3)
        file_name = video + "_frame_" + str(frame) + ".jpg"
        file_path = os.path.join(output_dir, file_name)
        scipy.misc.toimage(image[:,:,::-1]).save(file_path)


def proto_to_np(bbox, index):
    return np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2, index])


def main():
    dataset_name = 'kcam'
    job_name = 'composite_job'
    column = 'composite_bboxes'

    def load_and_convert_bboxes(job):
        bboxes = list(load_bboxes(dataset_name, job).as_frame_list())
        bboxes = {k: [vv for (_, vv) in v] for (k, v) in bboxes}
        return bboxes

    # bboxes=  load_and_convert_bboxes('track')
    # print bboxes
    # visualize_frames(dataset_name, '0', range(10000, 10100), {'0':bboxes}, 'imgs')
    # exit()

    start = now()
    write('Loading Facenet... ')
    facenet_bboxes = load_and_convert_bboxes('facenet_0_25')
    for vid in facenet_bboxes:
        for i in range(len(facenet_bboxes[vid])):
            filtered = [proto_to_np(x, 0) for x in facenet_bboxes[vid][i]]
            facenet_bboxes[vid][i] = filtered
    write_timer(start)

    start = now()
    write('Loading Faster-RCNN... ')
    frcnn_bboxes = load_and_convert_bboxes('frcnn')
    for vid in frcnn_bboxes:
        for i in range(len(frcnn_bboxes[vid])):
            filtered = [proto_to_np(x, 1) for x in frcnn_bboxes[vid][i] if x.label == PERSON]
            frcnn_bboxes[vid][i] = filtered
    write_timer(start)

    start = now()
    write('Loading CPM... ')
    cpm_bboxes = {}
    cpm2_data = load_cpm2_joint_centers(dataset_name, 'joints')
    _, poses_by_video = parse_cpm2_data(cpm2_data)
    for vid, poses_by_frame in poses_by_video.iteritems():
        cpm_bboxes[vid] = []
        for frame, poses in poses_by_frame.iteritems():
            frame_bboxes = []
            for pose in poses:
                pose = pose['pose']
                head = pose[0:2, 0:2]
                axis = head[1,:] - head[0,:]
                cross = [-axis[1], axis[0]]
                if np.abs(LA.norm(cross)) < 0.01 : continue
                cross /= LA.norm(cross)
                midpoint = [(head[0,0]+head[1,0])/2.0, (head[0,1]+head[1,1])/2.0]
                halfnorm = LA.norm(axis) / 2.0
                pts = [head[0,:], head[1,:]]
                pts.append(midpoint + cross * halfnorm)
                pts.append(midpoint - cross * halfnorm)
                pts = np.array(pts)
                bbox = np.array([np.min(pts[:,1]), np.min(pts[:,0]),
                                 np.max(pts[:,1]), np.max(pts[:,0]), 2])
                frame_bboxes.append(bbox)
            cpm_bboxes[vid].append(frame_bboxes)
    write_timer(start)

    start = now()
    write('Voting... ')
    output_bboxes = {}
    rows = {}
    for vid in cpm_bboxes:
        output_bboxes[vid] = []
        num_frames = len(cpm_bboxes[vid])
        rows[vid] = [i * 24 for i in range(num_frames)]
        for frame in range(num_frames):
            frame_bboxes = []
            # insert voting scheme
            frame_bboxes += frcnn_bboxes[vid][frame]
            frame_bboxes += facenet_bboxes[vid][frame]
            frame_bboxes += cpm_bboxes[vid][frame]
            # frame_bboxes = nms(frame_bboxes, 0.3)
            output_bboxes[vid].append(frame_bboxes)
    write_timer(start)

    start = now()
    write('Visualizing... ')
    visualize_frames(dataset_name, '2', rows['2'], output_bboxes, 'imgs')
    write_timer(start)
    # def serialize(l):
    #     output = struct.pack("=Q", len(l))
    #     for box in l:
    #         bbox = types.BoundingBox()
    #         bbox.x1 = box[0]
    #         bbox.y1 = box[1]
    #         bbox.x2 = box[2]
    #         bbox.y2 = box[3]
    #         s = bbox.SerializeToString()
    #         output += struct.pack("=i", len(s))
    #         output += s
    #     return output
    # db.write_job_result(dataset_name, job_name, column, serialize, rows, output_bboxes)


if __name__ == "__main__":
    main()
