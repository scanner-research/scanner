import numpy as np
import struct
import sys
import scanner

db = scanner.Scanner()

import scannerpy.evaluators.types_pb2 as evaluators

@db.loader('histogram')
def load_histograms(buf, metadata):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)

@db.loader('faces')
def load_faces(buf, metadata):
    num_faces = len(buf) / 16
    faces = []
    for i in range(num_faces):
        faces.append(struct.unpack("iiii", buf[(16*i):(16*(i+1))]))
    return faces

@db.loader('opticalflow')
def load_opticalflow(buf, metadata):
    return np.frombuffer(buf, dtype=np.dtype(np.float32)).reshape((metadata.width, metadata.height, 2))

@db.loader('cameramotion')
def load_cameramotion(buf, metadata):
    return struct.unpack('d', buf)

@db.loader('fc25')
def load_yolo_features(buf, metadata):
    return np.frombuffer(buf, dtype=np.dtype(np.float32))

@db.loader('pool10')
def load_squeezenet_features(buf, metadata):
    return np.frombuffer(buf, dtype=np.dtype(np.float32))

@db.loader('fc6')
def load_mscnn_features(buf, metadata):
    return np.frombuffer(buf, dtype=np.dtype(np.float32))

@db.loader('fc7')
def load_faster_rcnn_features(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    return np.split(buf, len(buf) / 4096) if len(buf) != 0 else np.array([])

@db.loader('bboxes')
def load_bboxes(buf, metadata):
    try:
        (num_bboxes,) = struct.unpack("=Q", buf[:8])
        buf = buf[8:]
        bboxes = []
        for i in range(num_bboxes):
            bbox_size, = struct.unpack("=i", buf[:4])
            buf = buf[4:]
            box = evaluators.BoundingBox()
            box.ParseFromString(buf[:bbox_size])
            bboxes.append(box)
            buf = buf[bbox_size:]
        return bboxes
    except struct.error:
        return []

@db.loader('frame')
def load_frames(buf, metadata):
    return np.frombuffer(buf, dtype=np.dtype(np.uint8)) \
             .resize((metadata.height, metadata.width, 3))

@db.loader('median')
def load_medians(buf, metadata):
    return np.frombuffer(buf, dtype=np.dtype(np.uint8))

cv_version = 3 # int(cv2.__version__.split('.')[0])

def save_movie_info():
    np.save('{}_faces.npy'.format(JOB), load_faces(JOB)[0]['buffers'])
    np.save('{}_histograms.npy'.format(JOB), load_histograms(JOB)[0]['buffers'])
    np.save('{}_opticalflow.npy'.format(JOB), load_opticalflow(JOB)[0]['buffers'])

# After running this, run:
# ffmpeg -safe 0 -f concat -i <(for f in ./*.mkv; do echo "file '$PWD/$f'"; done) -c copy output.mkv
def save_debug_video():
    bufs = load_output_buffers(JOB, 'video', lambda buf: buf)[0]['buffers']
    i = 0
    for buf in bufs:
        if len(buf) == 0: continue
        ext = 'mkv' if cv_version >= 3 else 'avi'
        with open('out_{:06d}.{}'.format(i, ext), 'wb') as f:
            f.write(buf)
        i += 1

def main():
    DATASET = sys.argv[1]
    JOB = sys.argv[2]

    with open('/bigdata/dfouhey/bboxes.txt', 'w') as f:
        for (vid, vid_bboxes) in list(load_bboxes(DATASET, JOB).as_frame_list()):
            for (frame, frame_bboxes) in vid_bboxes:
                for bbox in frame_bboxes:
                    s = ' '.join(map(str, [vid, frame, bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label]))
                    f.write(s + '\n')

    # np.save('{}_histograms.npy'.format(DATASET),
    #         [frame
    #          for (_, vid) in load_histograms(DATASET, JOB).as_frame_list()
    #          for (_, frame) in vid
    #         ])

if __name__ == "__main__":
    main()
