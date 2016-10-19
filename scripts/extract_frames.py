from __future__ import print_function
import scanner
from PIL import Image, ImageDraw
import imageio
import argparse
import os.path as path


def extract_frames(video_paths, indices_per_video, output_directory,
                   fn=lambda x, y, z: x):
    for vi, (video_path, frame_indices) in enumerate(
            zip(video_paths, indices_per_video)):
        video = imageio.get_reader(video_path, 'ffmpeg')
        video_name = path.splitext(path.basename(video_path))[0]
        for i, fi in enumerate(frame_indices):
            image = fn(Image.fromarray(video.get_data(fi)), vi, i)
            file_name = video_name + "_frame_" + str(fi) + ".jpg"
            file_path = path.join(output_directory, file_name)
            image.save(file_path, "JPEG")


def main(args):
    # Read list of video paths
    with open(args.video_paths, 'r') as f:
        video_paths = [line.strip() for line in f]
    # Read list of (video_index, frame_index) pairs
    with open(args.frame_indices, 'r') as f:
        frame_indices = [map(lambda x: int(x), line.strip().split(' '))
                         for line in f]
    # Read list of [(x1, y1, x2, y2), ...] lists
    with open(args.frame_bboxes, 'r') as f:
        frame_bboxes = [[map(lambda x: int(float(x)), p.strip().split(' '))
                         for p in line.split(',')]
                        for line in f]
        print(frame_bboxes)

    # Process into per video frame lists so we can extract sequentially
    indices_per_video = [[] for x in range(len(video_paths))]
    bboxes_per_video = [[] for x in range(len(video_paths))]
    for (vi, fi), bboxes in zip(frame_indices, frame_bboxes):
        assert(vi < len(video_paths))
        indices_per_video[vi].append(fi)
        bboxes_per_video[vi].append(bboxes)

    #extract_frames(video_paths, indices_per_video)
    def draw_bboxes(image, vi, frame_candidate):
        draw = ImageDraw.Draw(image)
        print(vi, frame_candidate)
        for bbox in bboxes_per_video[vi][frame_candidate]:
            draw.rectangle(bbox)
        del draw
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
