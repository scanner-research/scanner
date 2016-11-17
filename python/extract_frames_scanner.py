import argparse
import scanner
import numpy as np
import cv2
from decode import db

@db.loader('frame')
def load_frames(buf, metadata):
    return np.frombuffer(buf, dtype=np.uint8) \
             .reshape((metadata.height,metadata.width,3))


def write_indices(indices):
    d = {}
    for (vid, frame) in indices:
        if not vid in d: d[vid] = []
        d[vid].append(frame)
    with open('indices.txt', 'w') as f:
        for k, v in d.iteritems():
            s = '{} {}\n'.format(k, ' '.join(map(str,v)))
            f.write(s)

def extract_frames(dataset):
    success, _ = db.run(dataset, 'base', 'frame_extract', 'extracted',
                        {'force': True})
    if not success:
        print('Scanner failed')
        exit()

    for (vid, frames) in load_frames(dataset, 'extracted').as_frame_list():
        for (frame, buf) in frames:
            img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            cv2.imwrite('/bigdata/{}_{}.jpg'.format(vid, frame), img)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract JPEG frames from videos')
    p.add_argument('dataset', type=str)
    args = p.parse_args()
    extract_frames(args.dataset)
