import struct


def bboxes(buf, protobufs):
    s = struct.pack('=Q', len(buf))
    for bbox in buf:
        bs = bbox.SerializeToString()
        s += struct.pack('=Q', len(bs))
        s += bs
    return s


def poses(poses, protobufs):
    if len(poses) == 0:
        return b' '
    else:
        return b''.join([pose.keypoints.tobytes() for pose in poses])
