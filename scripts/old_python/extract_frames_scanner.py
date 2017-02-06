import argparse
import scanner
import numpy as np
import cv2
from decode import db
import os

@db.loader('frame')
def load_frames(buf, metadata):
    metadata = metadata[0]
    return np.frombuffer(buf, dtype=np.uint8) \
             .reshape((metadata.height,metadata.width,3))


def write_indices(indices):
    d = {}
    for (vid, frames) in indices:
        if not vid in d: d[vid] = []
        d[vid] += frames
    with open('indices.txt', 'w') as f:
        for k, v in d.iteritems():
            s = '{} {}\n'.format(k, ' '.join(map(str,v)))
            f.write(s)


def get_frames(dataset, frames, directory):
    write_indices(frames)
    success, _ = db.run(dataset, 'frame_extract', 'extracted',
                        {'force': True,
                         'io_item_size': 4096,
                         'work_item_size': 512,
                         'pus_per_node': 1})
    if not success:
        print('Scanner failed')
        exit()

    os.system('mkdir -p {}'.format(directory))
    for (vid, frames) in load_frames(dataset, 'extracted').as_frame_list():
        for (frame, buf) in frames:
            img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            path = '{}/{}_{:07d}.jpg'.format(directory, vid, frame)
            if not cv2.imwrite(path, img):
                print('imwrite failed')
                exit()


def extract_frames(args):
    success, _ = db.run(args['dataset'], 'frame_extract', 'extracted',
                        {'force': True,
                         'pus_per_node': args['pus_per_node']
                         if 'pus_per_node' in args else '1'})
    if not success:
        print('Scanner failed')
        exit()

    for (vid, frames) in load_frames(args['dataset'], 'extracted').as_frame_list():
        for (frame, buf) in frames:
            img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            path = '{}/{}_{:07d}.jpg'.format(args['out_dir'], vid, frame)
            if not cv2.imwrite(path, img):
                print('imwrite failed')
                exit()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract JPEG frames from videos')
    p.add_argument('dataset', type=str)
    p.add_argument('out_dir', type=str)
    extract_frames(p.parse_args().__dict__)
