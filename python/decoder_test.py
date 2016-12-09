import argparse
import scanner
import numpy as np
import cv2
from decode import db

@db.loader('frame')
def load_frames(buf, metadata):
    return np.frombuffer(buf, dtype=np.uint8) \
             .reshape((metadata.height,metadata.width,3))

def extract_frames(args):
    job = load_frames(args['dataset'], 'edr')
    video_paths = job._dataset.video_data.original_video_paths
    for (vid, frames) in job.as_frame_list():
        video_path = video_paths[int(vid)]
        inp = cv2.VideoCapture(video_path)
        assert(inp.isOpened())
        video_frame_num = -1
        for (frame_num, buf) in frames:
            while video_frame_num != frame_num:
                _, video_frame = inp.read()
                video_frame_num += 1
            scanner_frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            frame_diff = (scanner_frame - video_frame).sum()
            if frame_diff != 0:
                print('Frame {} does not match!'.format(frame_num))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract JPEG frames from videos')
    p.add_argument('dataset', type=str)
    extract_frames(p.parse_args().__dict__)
