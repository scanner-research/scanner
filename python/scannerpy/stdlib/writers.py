import struct

def bboxes(bufs):
    s = struct.pack('=Q', len(bufs[0]))
    for bbox in bufs[0]:
        bs = bbox.SerializeToString()
        s += struct.pack('=Q', len(bs))
        s += bs
    return [s]
