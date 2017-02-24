import numpy as np
import cv2
import struct


def bboxes(buf, db):
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


def histograms(buf, db):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)


def frame_info(buf, db):
    info = db.protobufs.FrameInfo()
    return info


def flow((buf_flow, buf_frame_info), db):
    info = frame_info(buf_frame_info, db)
    output = np.formbuffer(buf, dtype=np.dtype(np.float32))
    output.reshape((info.height, info.width, 2))
    return output
