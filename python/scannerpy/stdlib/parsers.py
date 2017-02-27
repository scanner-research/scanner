import numpy as np
import cv2
import struct


def bboxes(bufs, db):
    buf = bufs[0]
    (num_bboxes,) = struct.unpack("=Q", buf[:8])
    buf = buf[8:]
    bboxes = []
    for i in range(num_bboxes):
        (bbox_size,) = struct.unpack("=i", buf[:4])
        buf = buf[4:]
        box = db.protobufs.BoundingBox()
        box.ParseFromString(buf[:bbox_size])
        buf = buf[bbox_size:]
        bbox = [box.x1, box.y1, box.x2, box.y2, box.score,
                box.track_id, box.track_score]
        bboxes.append(bbox)
    return bboxes


def histograms(bufs, db):
    return np.split(np.frombuffer(bufs[0], dtype=np.dtype(np.int32)), 3)


def frame_info(buf, db):
    info = db.protobufs.FrameInfo()
    info.ParseFromString(buf)
    return info


def flow(bufs, db):
    output = np.frombuffer(bufs[0], dtype=np.dtype(np.float32))
    info = frame_info(bufs[1], db)
    return output.reshape((info.height, info.width, 2))
