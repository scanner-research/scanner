from __future__ import print_function
import json
import numpy as np
import struct
import sys
from pprint import pprint

DB_PATH = '/disk0/apoms/scanner_db'

def read_string(f):
    s = ""
    while True:
        byts = f.read(1)
        (c,) = struct.unpack("c", byts)
        if c == '\0':
            break
        s += c
    return s


def load_db_metadata():
    path = DB_PATH + '/db_metadata.bin'
    meta = {}
    with open(path, 'rb') as f:
        byts = f.read(4)
        (next_dataset_id,) = struct.unpack("=i", byts)
        meta["next_dataset_id"] = next_dataset_id

        byts = f.read(4)
        (next_job_id,) = struct.unpack("=i", byts)
        meta["next_job_id"] = next_job_id

        byts = f.read(8)
        (num_datasets,) = struct.unpack("=Q", byts)
        dataset_names = {} 
        for i in range(num_datasets):
            byts = f.read(4)
            (dataset_id,) = struct.unpack("=i", byts)
            s = read_string(f)
            dataset_names[dataset_id] = s
        meta["dataset_names"] = dataset_names

        dataset_job_ids = {}
        for i in range(num_datasets):
            byts = f.read(4)
            (dataset_id,) = struct.unpack("=i", byts)
            byts = f.read(8)
            (num_job_ids,) = struct.unpack("=Q", byts)
            dataset_job_ids[dataset_id] = []
            for j in range(num_job_ids):
                byts = f.read(4)
                (job_id,) = struct.unpack("=i", byts)
                dataset_job_ids[dataset_id].append(job_id)
        meta["dataset_job_ids"] = dataset_job_ids

        byts = f.read(8)
        (num_jobs,) = struct.unpack("=Q", byts)
        job_names = {}
        for i in range(num_jobs):
            byts = f.read(4)
            (job_id,) = struct.unpack("=i", byts)
            s = read_string(f)
            job_names[job_id] = s
        meta["job_names"] = job_names
    return meta


def write_db_metadata(meta):
    path = DB_PATH + '/db_metadata.bin'
    with open(path, 'wb') as f:
        f.write(struct.pack("=iiQ",
                            meta["next_dataset_id"],
                            meta["next_job_id"],
                            len(meta["dataset_names"])))
        for i, name in meta["dataset_names"].iteritems():
            f.write(struct.pack("=i", i))
            for c in name:
                f.write(struct.pack("=c", c))
            f.write(struct.pack("=c", '\0'))

        for i, name in meta["dataset_names"].iteritems():
            d_job_ids = meta["dataset_job_ids"][i]
            f.write(struct.pack("=iQ", i, len(d_job_ids)))
            for job_id in d_job_ids:
                f.write(struct.pack("=i", job_id))

        f.write(struct.pack("=Q", len(meta["job_names"])))
        for i, name in meta["job_names"].iteritems():
            f.write(struct.pack("=i", i))
            for c in name:
                f.write(struct.pack('=c', c))
            f.write(struct.pack('=c', '\0'))


def load_output_buffers(job_name, column, fn, intvl=None):
    with open(DB_PATH + '/{}_job_descriptor.bin'.format(job_name), 'r') as f:
        job = json.loads(f.read())

    videos = []
    for json_video in job['videos']:
        video = {'path': json_video['path'], 'buffers': []}
        (istart, iend) = intvl if intvl is not None else (0, sys.maxint)
        for [start, end] in json_video['intervals']:
            path = DB_PATH + '/{}_job/{}_{}_{}-{}.bin'.format(
                job_name, json_video['path'], column, start, end)
            if start > iend or end < istart: continue
            try:
                with open(path, 'rb') as f:
                    lens = []
                    start_pos = sys.maxint
                    pos = 0
                    for i in range(end-start):
                        idx = i + start
                        byts = f.read(8)
                        (buf_len,) = struct.unpack("l", byts)
                        old_pos = pos
                        pos += buf_len
                        if (idx >= istart and idx <= iend):
                            if start_pos == sys.maxint:
                                start_pos = old_pos
                            lens.append(buf_len)

                    bufs = []
                    f.seek((end-start) * 8 + start_pos)
                    for buf_len in lens:
                        buf = f.read(buf_len)
                        item = fn(buf)
                        bufs.append(item)

                    video['buffers'] += bufs
            except IOError as e:
                print("{}".format(e))
        videos.append(video)

    return videos


def write_output_buffers(dataset_name, job_name, column, fn, data):
    start = 0
    end = len(data)

    job = {
        'dataset_name': dataset_name,
        'videos': [
            {
                'intervals': [[start, end]],
                'path': '0'
            }
        ],
    }

    path = DB_PATH + '/{}_job/{}_{}_{}-{}.bin'.format(
        job_name, '0', column, start, end)

    with open(path, 'wb') as f:
        all_bytes = ""
        for d in data:
            byts = fn(d)
            all_bytes += byts
            f.write(struct.pack("=Q", len(byts)))
        f.write(all_bytes)

    with open(DB_PATH + '/{}_job_descriptor.bin'.format(job_name), 'w') as f:
        f.write(json.dumps(job))


def get_output_size(job_name):
    with open(DB_PATH + '/{}_job_descriptor.bin'.format(job_name), 'r') as f:
        job = json.loads(f.read())

    return job['videos'][0]['intervals'][-1][1]

def load_histograms(job_name):
    def buf_to_histogram(buf):
        return np.split(np.frombuffer(buf, dtype=np.dtype(np.float32)), 3)
    return load_output_buffers(job_name, 'histogram', buf_to_histogram)

def load_faces(job_name):
    def buf_to_faces(buf):
        num_faces = len(buf) / 16
        faces = []
        for i in range(num_faces):
            faces.append(struct.unpack("iiii", buf[(16*i):(16*(i+1))]))
        return faces
    return load_output_buffers(job_name, 'faces', buf_to_faces)

JOB = 'star'

def save_movie_info():
    np.save('{}_faces.npy'.format(JOB), load_faces(JOB)[0]['buffers'])
    np.save('{}_histograms.npy'.format(JOB), load_histograms(JOB)[0]['buffers'])

# After running this, run:
# ffmpeg -safe 0 -f concat -i <(for f in ./*.mkv; do echo "file '$PWD/$f'"; done) -c copy output.mkv
def save_debug_video():
    bufs = load_output_buffers(JOB, 'video', lambda buf: buf)[0]['buffers']
    i = 0
    for buf in bufs:
        if len(buf) == 0: continue
        with open('out__{:06d}.mkv'.format(i), 'wb') as f:
            f.write(buf)
        i += 1


def load_bboxes(job_name, column_name):
    def buf_to_bboxes(buf):
        (num_bboxes,) = struct.unpack("=Q", buf[:8])
        buf = buf[8:]
        bboxes = []
        for i in range(num_bboxes):
            bboxes.append(struct.unpack("fffff", buf[(20*i):(20*(i+1))]))
        return bboxes
    return load_output_buffers(job_name, column_name, buf_to_bboxes)


def save_bboxes(dataset_name, job_name, column_name, bboxes):
    def bboxes_to_buf(boxes):
        data = struct.pack("=Q", len(boxes))
        for i in range(len(boxes)):
            data += struct.pack("=fffff",
                                boxes[i][0],
                                boxes[i][1],
                                boxes[i][2],
                                boxes[i][3],
                                boxes[i][4])
        return data
    return write_output_buffers(dataset_name, job_name, column_name,
                                bboxes_to_buf, bboxes)


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
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
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
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def collate_boxes(base, tr):
    gold = [[] for i in range(len(base))]
    tracks = []

    avg = 0
    for i in range(len(base)):
        base_boxes = base[i]
        tracked_boxes = tr[i]

        avg += len(base_boxes)
        avg += len(tracked_boxes)

        for box in base_boxes:
            # check for overlap
            overlapped = False
            for t in tracks:
                if iou(box, t['boxes'][-1]) > 0.2:
                    t['last_detection'] = 0
                    t['total_detections'] += 1
                    t['boxes'].append(box)
                    overlapped = True
                    break
            if overlapped:
                continue
            # else add new track if threshold high enough
            if box[4] > 3:
                track = {
                    'start': i,
                    'start_box': box,
                    'boxes': [box],
                    'last_detection': 0,
                    'total_detections': 0,
                }
                tracks.append(track)

        dead_tracks = []
        for z, t in enumerate(tracks):
            t['last_detection'] += 1
            if t['last_detection'] > 10:
                # remove track
                if t['total_detections'] > 8:
                    for n in range(0, i - t['start']):
                        gold[t['start'] + n].append(t['boxes'][n])
                dead_tracks.append(z)
                continue

            if len(t['boxes']) > i: continue
            overlapped = False
            for box in tracked_boxes:
                # check for overlap
                if iou(box, t['boxes'][-1]) > 0.2:
                    t['boxes'].append(box)
                    t['total_detections'] += 1
                    overlapped = True
                    break
            if not overlapped:
                t['boxes'].append(t['boxes'][-1])
                print('DID NOT OVERLAP')

        dead_tracks.reverse()
        for z in dead_tracks:
            tracks.pop(z)

    gold_avg = 0
    for boxes in gold:
        gold_avg += len(boxes)

    gold_avg /= len(base)
    avg /= len(base)
    print("gold", gold_avg)
    print("avg", avg)

    return gold



def main():
    #save_debug_video()
    meta = load_db_metadata()
    pprint(meta)
    data = load_bboxes("facenet_eating_bg", "base_bboxes")
    base_boxes = data[0]["buffers"]
    data = load_bboxes("facenet_eating_bg", "tracked_bboxes")
    tracked_boxes = data[0]["buffers"]
    gold_boxes = collate_boxes(base_boxes, tracked_boxes)
    save_bboxes("eating_test", "bboxes_test", "base_bboxes", gold_boxes)
    job_id = meta["next_job_id"]
    meta["next_job_id"] += 1
    meta["job_names"][job_id] = "bboxes_test"
    dataset_id = -1
    for i, name in meta["dataset_names"].iteritems():
        if name == "eating_test":
            dataset_id = i
            break;
    assert(dataset_id != -1)
    meta["dataset_job_ids"][dataset_id].append(job_id)
    write_db_metadata(meta)

if __name__ == "__main__":
    main()
