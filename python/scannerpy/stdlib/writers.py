import struct

def bboxes(bufs, protobufs):
    s = struct.pack('=Q', len(bufs[0]))
    for bbox in bufs[0]:
        bs = bbox.SerializeToString()
        s += struct.pack('=Q', len(bs))
        s += bs
    return [s]

def poses(bufs, protobufs):
    s = struct.pack("=Q", len(bufs[0]))
    for pose in bufs[0]:
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
    return [s]
