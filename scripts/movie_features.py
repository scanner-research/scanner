import scanner
from collections import defaultdict
from pprint import pprint
import struct
import numpy as np
import scipy.misc
import cv2
import os.path
import extract_frames_scanner
import subprocess
import yaml
import requests
import json
import sys
from timeit import default_timer as now

db = scanner.Scanner()
dataset_name = 'movie'
from scannerpy import metadata_pb2
from scannerpy.evaluators import types_pb2


def write(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def write_timer(start):
    write('{:.3f}\n'.format(now() - start))


@db.loader('histogram')
def load_histograms(buf, config):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)


@db.loader('landmarks')
def load_frames(buf, metadata):
    metadata = metadata[0]
    return np.frombuffer(buf, dtype=np.uint8) \
             .reshape((metadata.height,metadata.width,3))


def load_bboxes(dataset_name, job_name, column_name):
    def buf_to_bboxes(buf, md):
        (num_bboxes,) = struct.unpack("=Q", buf[:8])
        buf = buf[8:]
        bboxes = []
        for i in range(num_bboxes):
            (bbox_size,) = struct.unpack("=i", buf[:4])
            buf = buf[4:]
            box = types_pb2.BoundingBox()
            box.ParseFromString(buf[:bbox_size])
            buf = buf[bbox_size:]
            bbox = [box.x1, box.y1, box.x2, box.y2, box.score,
                    box.track_id, box.track_score]
            bboxes.append(bbox)
        return bboxes
    return db.get_job_result(dataset_name, job_name, column_name,
                             buf_to_bboxes)


def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    elif len(boxes) == 1:
        return boxes

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
    # Extract frames
    frames = [(video, v_frames)]
    print 'Extracting frames'
    extract_frames_scanner.get_frames(dataset_name, frames,
                                      '/tmp/scanner_frames')
    image_template = '/tmp/scanner_frames/{}_{:07d}.jpg'
    print 'Drawing bboxes'
    for i, frame in enumerate(v_frames):
        image = cv2.imread(image_template.format(video, frame))
        #r, image = cap.read()
        boxes = bboxes[i]
        for bbox in boxes:
            bbox = np.array(bbox).astype(int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 0, 255), 3)
        file_name = video + "_frame_" + str(frame) + ".jpg"
        file_path = os.path.join(output_dir, file_name)
        scipy.misc.toimage(image[:,:,::-1]).save(file_path)


def get_dimensions(path):
    cmd = "ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 {}"
    s = subprocess.check_output(cmd.format(path), shell=True).split("\n")
    width = float(s[0].split("=")[1])
    height = float(s[1].split("=")[1])
    return width, height


def get_movie_metadata(title):
    r = requests.get('http://www.omdbapi.com', params={
        't': title
    })
    data = r.json()
    return data


def serialize_bboxes(l):
    output = struct.pack("=Q", len(l))
    for box in l:
        bbox = types_pb2.BoundingBox()
        bbox.x1 = box[0]
        bbox.y1 = box[1]
        bbox.x2 = box[2]
        bbox.y2 = box[3]
        s = bbox.SerializeToString()
        output += struct.pack("=i", len(s))
        output += s
    return output


def main():
    movies = [
        ('/n/scanner/wcrichto.new/videos/meanGirls_short.mp4',
         'Mean Girls'),
        # ('/n/scanner/wcrichto.new/videos/charade_short.mkv',
        #  'Charade'),
        # ('/n/scanner/wcrichto.new/videos/movies/private/meanGirls.mp4',
        #  'Mean Girls'),
    ]
    dataset_name = 'movie_features'
    hosts = ['ocean.pdl.local.cmu.edu', 'crissy.pdl.local.cmu.edu']
    opts = {
        'force': True,
        'hosts': hosts,
        'node_count': len(hosts),
        'pus_per_node': 4,
        'env': {}
    }

    for (vid, frames) in load_frames(dataset_name, 'landmarks').as_frame_list():
        for (frame, buf) in frames:
            img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            path = '{}/{}_{:07d}.jpg'.format('/tmp/scanner', vid, frame)
            if not cv2.imwrite(path, img):
                print('imwrite failed')
                exit()
    exit()

    start = now()
    write('Retrieving movie metadata... ')
    # TODO: where to write metadata?
    all_metadata = []
    for path, title in movies:
        metadata = get_movie_metadata(title)
        (width, height) = get_dimensions(path)
        metadata['width'] = width
        metadata['height'] = height
        all_metadata.append(metadata)
    write_timer(start)

    # start = now()
    # write('Ingesting... ')
    # db.ingest('video', dataset_name, [path for path, _ in movies],
    #           {'force': True})
    # write_timer(start)

    def shot_boundaries():
        fg_dir = '/h/wcrichto/mp/film_grammar_lite/python/fg_pipeline'
        result, t = db.run(dataset_name, 'histogram', 'hist', opts)

        hists = load_histograms(dataset_name, 'hist').as_frame_list()

        # TODO: where to write the shots.txt?
        for (_, vid) in hists:
            np.save(
                '{}/tmp/histograms.npy'.format(fg_dir),
                [frame for (_, frame) in vid])
            subprocess.check_call(
                'cd {}; python process_movie.py tmp'.format(fg_dir), shell=True)
            subprocess.check_call(
                'mv {}/tmp/shots.txt /tmp/shots.txt'.format(fg_dir), shell=True)

    def face_bboxes():
        scales = [0.125, 0.25, 0.5, 1]
        job_names = ['face_{}'.format(scale) for scale in scales]

        # for (scale, job_name) in zip(scales, job_names):
        #     opts['env']['SC_SCALE'] = '{}'.format(scale)
        #     rc, t = db.run(dataset_name, 'facenet', job_name, opts)

        all_bboxes = defaultdict(list)
        for job_name in job_names:
            boxes = load_bboxes(dataset_name, job_name, "base_bboxes").as_outputs()
            for data in boxes:
                all_bboxes[data['table']].append(data['buffers'])

        nms_bboxes = defaultdict(list)
        for vi, boxes in all_bboxes.iteritems():
            frames = len(boxes[0])
            runs = len(boxes)
            new_boxes = []
            for fi in range(frames):
                frame_boxes = []
                for r in range(runs):
                    frame_boxes += (boxes[r][fi])
                frame_boxes = nms(frame_boxes, 0.3)
                new_boxes.append(frame_boxes)
            nms_bboxes[vi] = new_boxes

        # Write bounding boxes back as a job
        lens = db.get_job_result(dataset_name, 'base', 'w/e', lambda x: x) \
          .get_table_lengths()
        output_bboxes = defaultdict(list)
        rows = defaultdict(list)
        stride = 24
        for vid in nms_bboxes:
            num_frames = len(nms_bboxes[vid])
            expected = lens[vid]
            rows[vid] = list(range(max(num_frames * stride, expected)))
            for i in range(num_frames):
                output_bboxes[vid].append(nms_bboxes[vid][i])
                for _ in range(stride-1):
                    output_bboxes[vid].append([])
            for _ in range(len(output_bboxes[vid]), expected):
                output_bboxes[vid].append([])
            rows[vid] = rows[vid][:expected]
            output_bboxes[vid] = output_bboxes[vid][:expected]
            assert expected == len(rows[vid])
            assert expected == len(output_bboxes[vid])

        db.write_job_result(dataset_name, 'composite_job', 'composite_bboxes',
                            serialize_bboxes, rows, output_bboxes)

        snarf = []
        for vi in nms_bboxes.keys():
            meta = all_metadata[int(vi)]
            snarf_vid = {}
            snarf_vid['title'] = str(meta['Title'])
            snarf_vid['frames'] = []

            width = meta['width']
            height = meta['height']

            for (i, frame_bboxes) in enumerate(nms_bboxes['0']):
                snarf_frame = {
                    'frame_number': '{:06d}'.format(i * 24),
                    'faces': []
                }

                for bbox in frame_bboxes:
                    snarf_bbox = {
                        'x': float(bbox[0]/width),
                        'y': float(bbox[1]/height),
                        'w': float((bbox[2] - bbox[0])/width),
                        'h': float((bbox[3] - bbox[1])/height),
                        'confidence': float(bbox[4])
                    }
                    snarf_frame['faces'].append(snarf_bbox)

                snarf_vid['frames'].append(snarf_frame)
            snarf.append(snarf_vid)

        with open('/tmp/faces.yml', 'w') as f:
            f.write(yaml.dump(snarf))

    # start = now()
    # write('Computing shot boundaries... ')
    # shot_boundaries()
    # write_timer(start)

    start = now()
    write('Computing face bboxes... ')
    face_bboxes()
    write_timer(start)


if __name__ == "__main__":
    main()
