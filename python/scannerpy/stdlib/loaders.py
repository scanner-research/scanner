from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2
import struct

def bboxes(db, buf):
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

def histograms(buf):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)
