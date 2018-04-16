from __future__ import absolute_import, division, print_function, unicode_literals
import struct


def bboxes(buf, protobufs):
    s = struct.pack('=Q', len(buf))
    for bbox in buf:
        bs = bbox.SerializeToString()
        s += struct.pack('=Q', len(bs))
        s += bs
    return s


def poses(buf, protobufs):
    s = struct.pack("=Q", len(buf))
    for pose in buf:
        # Num joints
        s += struct.pack("=Q", len(pose))
        for i in range(len(pose)):
            point = protobufs.Point()
            point.y = pose[i, 0]
            point.x = pose[i, 1]
            point.score = pose[i, 2]
            # Point size
            s += struct.pack("=Q", point.ByteSize())
            s += point.SerializeToString()
    return s
