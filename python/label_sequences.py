from __future__ import print_function
import argparse
import os.path as path


def main(args):
    with open(args.video_paths, 'r') as f:
        video_paths = [line.strip() for line in f]
    # Read list of tracks
    with open(args.sequences, 'r') as f:
        sequence_bboxes = pickle.load(f)
        print(sequence_bboxes)
    frame_dir = args.frame_directory

    # For each track, display the track frames and wait for user to label
    labeled_sequences = [[] for i in range(len(video_paths))]
    labeled_frames = [[] for i in range(len(video_paths))]
    for video_paths, sequences in zip(video_paths, sequence_bboxes):


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description='Label sequences as positive or negative')
    p.add_argument('video_paths', type=str)
    p.add_argument('frame_directory', type=str)
    p.add_argument('sequences', type=str)
    p.add_argument('labeled_sequences', type=str)
    main()
