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
        bboxes.append(box)
    return bboxes


def poses(buf, db):
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
            point = db.protobufs.Point()
            point.ParseFromString(buf[:point_size])
            buf = buf[point_size:]
            joints[i, 0] = point.y
            joints[i, 1] = point.x
            joints[i, 2] = point.score
        bodies.append(joints)
    return bodies


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


def array(ty):
    def parser(bufs, db):
        return np.frombuffer(bufs[0], dtype=np.dtype(ty))
    return parser


def image(bufs, db):
    return cv2.imdecode(np.frombuffer(bufs[0], dtype=np.dtype(np.uint8)),
                        cv2.IMREAD_COLOR)

def raw_frame_gen(shape0, shape1, shape2, typ):
    def parser(bufs, db):
        output = np.frombuffer(bufs, dtype=typ)
        return output.reshape((shape0, shape1, shape2))
    return parser
