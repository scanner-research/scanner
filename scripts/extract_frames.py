from __future__ import print_function
import scanner
from PIL import Image
import imageio
import argparse
import os.path as path

def main(args):
    # Read list of video paths
    with open(args.video_paths, 'r') as f:
        video_paths = [line.strip() for line in f]
    # Read list of (video_index, frame_index) pairs
    with open(args.frame_indices, 'r') as f:
        frame_indices = [map(lambda x: int(x), line.strip().split(' '))
                         for line in f]
    # Process into per video frame lists so we can extract sequentially
    indices_per_video = [[] for x in range(len(video_paths))]
    for (vi, fi) in frame_indices:
        assert(vi < len(video_paths))
        indices_per_video[vi].append(fi)
    for i in indices_per_video:
        i.sort()

    # Extract frames
    for video_path, frame_indices in zip(video_paths, indices_per_video):
        video = imageio.get_reader(video_path, 'ffmpeg')
        video_name = path.splitext(path.basename(video_path))[0]
        for fi in frame_indices:
            image = Image.fromarray(video.get_data(fi))
            file_name = video_name + "_frame_" + str(fi) + ".jpg"
            file_path = path.join(args.output_directory, file_name)
            image.save(file_path, "JPEG")



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract JPEG frames from videos')
    p.add_argument('video_paths', type=str)
    p.add_argument('frame_indices', type=str)
    p.add_argument('output_directory', type=str)
    main(p.parse_args())
