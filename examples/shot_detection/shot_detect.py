from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
from scipy.spatial import distance
import numpy as np
import cv2
import math
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

try:
    import plotly.offline as offline
    import plotly.graph_objs as go
except ImportError:
    print('You need to install plotly to run this. Try running:\npip install plotly')
    exit()

WINDOW_SIZE = 500

def compute_shot_boundaries(hists):
    # Compute the mean difference between each pair of adjacent frames
    diffs = np.array([np.mean([distance.chebyshev(hists[i-1][j], hists[i][j])
                               for j in range(3)])
                      for i in range(1, len(hists))])
    diffs = np.insert(diffs, 0, 0)
    n = len(diffs)

    # Plot the differences. Look at histogram-diffs.html
    data = [go.Scatter(x=range(n),y=diffs)]
    offline.plot(data, filename='histogram-diffs.html')

    # Do simple outlier detection to find boundaries between shots
    boundaries = []
    for i in range(1, n):
        window = diffs[max(i-WINDOW_SIZE,0):min(i+WINDOW_SIZE,n)]
        if diffs[i] - np.mean(window) > 3 * np.std(window):
            boundaries.append(i)
    return boundaries


def make_montage(n, frames):
    _, frame = frames.next()
    frame = frame[0]
    (frame_h, frame_w, _) = frame.shape
    target_w = 256
    target_h = int(target_w / float(frame_w) * frame_h)
    frames_per_row = 8
    img_w = frames_per_row * target_w
    img_h = int(math.ceil(float(n) / frames_per_row)) * target_h
    img = np.zeros((img_h, img_w, 3))

    def place_image(i, fr):
        fr = cv2.resize(fr, (target_w, target_h))
        fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        row = i / frames_per_row
        col = i % frames_per_row
        img[(row * target_h):((row+1) * target_h),
            (col * target_w):((col+1) * target_w),
            :] = fr

    place_image(0, frame)
    for i, (_, frame) in enumerate(frames):
        place_image(i + 1, frame[0])

    return img


def make_montage_scanner(db, table, shot_starts):
    montage_args = db.protobufs.MontageArgs()
    montage_args.num_frames = len(shot_starts)
    montage_args.target_width = 256
    montage_args.frames_per_row = 8
    montage_op = db.ops.Montage(args=montage_args, device=DeviceType.GPU)
    selected_frames = [db.sampler().gather((table, 'mont'), shot_starts,
                                          item_size=len(shot_starts))]
    montage_collection = db.run(selected_frames, montage_op, 'montage_image',
                                force=True)
    for _, img in montage_collection.tables(0).load([0]):
        pass
    img = np.frombuffer(img[0], dtype=np.uint8)
    img = np.flip(np.reshape(img, (-1, 256 * 8, 3)), 2)
    return img


def main():
    movie_path = util.download_video() if len(sys.argv) <= 1 else sys.argv[1]
    print('Detecting shots in movie {}'.format(movie_path))
    movie_name = os.path.basename(movie_path)

    with Database() as db:
        if not db.has_table(movie_name):
            print('Loading movie into Scanner database...')
            db.ingest_videos([(movie_name, movie_path)], force=True)
        movie_table = db.table(movie_name)

        if not db.has_table(movie_name + '_hist'):
            print('Computing a color histogram for each frame...')
            db.run(
                db.sampler().all([(movie_table.name(), movie_name + '_hist')]),
                db.ops.Histogram(device=DeviceType.GPU),
                force=True)
        hists_table = db.table(movie_name + '_hist')

        print('Computing shot boundaries...')
        # Read histograms from disk
        hists = [h for _, h in hists_table.load(['histogram'], parsers.histograms)]
        boundaries = compute_shot_boundaries(hists)

        print('Visualizing shot boundaries...')
        # Make montage in scanner
        montage_img = make_montage_scanner(db, movie_table.name(), boundaries)

        # Loading the frames for each shot boundary
        # frames = movie_table.load([0], rows=boundaries)
        # montage_img = make_montage(len(boundaries), frames)

        cv2.imwrite('shots.jpg', montage_img)
        print('Successfully generated shots.jpg')

if __name__ == "__main__":
    main()
