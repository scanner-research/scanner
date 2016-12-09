from __future__ import print_function
import scanner
from PIL import Image, ImageDraw
import imageio
import argparse
import os.path as path
import cv2
import scipy.misc
import numpy as np


def extract_frames(video_paths, indices_per_video, output_directory,
                   fn=lambda x, y, z: x):
    for vi, (video_path, frame_indices) in enumerate(
            zip(video_paths, indices_per_video)):
        #cap = cv2.VideoCapture(video_path)
        video_name = path.splitext(path.basename(video_path))[0]
        #for i, fi in enumerate(frame_indices):
        #for i in range(18250):
        #    cap.read()
        for i, fi in enumerate(range(18250, 19150)):
            image_path = 'imgs/205310_836_frame_{}.jpg'.format(fi)
            #r, image = cap.read()
            image = cv2.imread(image_path)
            image = fn(image, vi, i)
            file_name = video_name + "_frame_" + str(fi) + ".jpg"
            file_path = path.join(output_directory, file_name)
            scipy.misc.toimage(image[:,:,::-1]).save(file_path)


def main(args):
    # Read list of video paths
    with open(args.video_paths, 'r') as f:
        video_paths = [line.strip() for line in f]
    # Read list of (video_index, frame_index) pairs
    with open(args.frame_indices, 'r') as f:
        frame_indices = [map(lambda x: int(x), line.strip().split(' '))
                         for line in f]
    # Read list of [(x1, y1, x2, y2), ...] lists
    with open('face_frame_bboxes.txt', 'r') as f:
        face_frame_bboxes = [[map(lambda x: int(float(x)), p.strip().split(' '))
                         for p in line.split(',')]
                        for line in f]

    with open('cpm_frame_bboxes.txt', 'r') as f:
        cpm_frame_bboxes = [[map(lambda x: int(float(x)), p.strip().split(' '))
                             for p in (line.split(',') if len(line) > 3 else [])]
                            for line in f]

    # Process into per video frame lists so we can extract sequentially
    indices_per_video = [[] for x in range(len(video_paths))]
    face_bboxes_per_video = [[] for x in range(len(video_paths))]
    for (vi, fi), face_bboxes in zip(frame_indices,
                                     face_frame_bboxes):
        print(vi)
        print(video_paths)
        assert(vi < len(video_paths))
        indices_per_video[vi].append(fi)
        face_bboxes_per_video[vi].append(face_bboxes)

    #extract_frames(video_paths, indices_per_video)
    last_index = [0]
    def draw_bboxes(image, vi, frame_candidate):
        print(vi, frame_candidate)
        indicies = indices_per_video[vi]
        if last_index[0] < len(indicies) and indicies[last_index[0]] == frame_candidate:
            for bbox in face_bboxes_per_video[vi][last_index[0]]:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 0, 255), 3)
            last_index[0] += 1
        for bbox in cpm_frame_bboxes[frame_candidate]:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 0), 3)
        return image

    extract_frames(video_paths, indices_per_video, args.output_directory,
                   draw_bboxes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract JPEG frames from videos')
    p.add_argument('video_paths', type=str)
    p.add_argument('frame_indices', type=str)
    p.add_argument('frame_bboxes', type=str)
    p.add_argument('output_directory', type=str)
    main(p.parse_args())
