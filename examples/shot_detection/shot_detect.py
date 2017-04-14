from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import parsers
from scipy.spatial import distance
import numpy as np
import cv2
import math
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util
import time

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
    target_w = 64
    target_h = int(target_w / float(frame_w) * frame_h)
    frames_per_row = 16
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
    row_length = 8
    rows_per_item = 1
    target_width = 256

    frame, frame_info = table.as_op().gather(
        shot_starts, item_size = row_length * rows_per_item)

    montage = db.ops.Montage(
        frame = frame,
        frame_info = frame_info,
        num_frames = row_length * rows_per_item,
        target_width = target_width,
        frames_per_row = row_length,
        device = DeviceType.GPU)

    job = Job(columns = [montage], name = 'montage_image')
    montage_table = db.run(job, force=True)

    montage_img = np.zeros((1, target_width * row_length, 3), dtype=np.uint8)
    for _, img in montage_table.load(['montage']):
        if len(img[0]) > 100:
            img = np.frombuffer(img[0], dtype=np.uint8)
            img = np.flip(np.reshape(img, (-1, target_width * row_length, 3)), 2)
            montage_img = np.vstack((montage_img, img))
    return montage_img


def main():
    movie_path = util.download_video() if len(sys.argv) <= 1 else sys.argv[1]
    print('Detecting shots in movie {}'.format(movie_path))
    movie_name = os.path.basename(movie_path)

    with Database(master='crissy.pdl.local.cmu.edu:5001',
                  workers=['crissy.pdl.local.cmu.edu:5002',
                           'stinson.pdl.local.cmu.edu:5002']) as db:
    # with Database(debug=True) as db:
        print('Loading movie into Scanner database...')
        s = time.time()
        [movie_table], _ = db.ingest_videos([(movie_name, movie_path)], force=True)
        print('Time: {:.1f}s'.format(time.time() - s))

        s = time.time()
        print('Computing a color histogram for each frame...')
<<<<<<< Updated upstream
        frame, frame_info = movie_table.as_op().all()
        histogram = db.ops.Histogram(
            frame = frame, frame_info = frame_info,
            device=DeviceType.GPU)
        job = Job(columns = [histogram], name = movie_name + '_hist')
        hists_table = db.run(job, force=True)
        print('\nTime: {:.1f}s'.format(time.time() - s))
=======
        db.run(
            db.sampler().all([(movie_table.name(), movie_name + '_hist')],
                             item_size=250),
            db.ops.Histogram(device=DeviceType.GPU),
            force=True)
        hists_table = db.table(movie_name + '_hist')
        print('')
        print('Time: {:.1f}s'.format(time.time() - s))
>>>>>>> Stashed changes

        s = time.time()
        print('Computing shot boundaries...')
        # Read histograms from disk
        hists = [h for _, h in hists_table.load(['histogram'],
                                                parsers.histograms)]
        boundaries = compute_shot_boundaries(hists)
        print('Time: {:.1f}s'.format(time.time() - s))

        s = time.time()
        print('Creating shot montage...')
        # Make montage in scanner
        montage_img = make_montage_scanner(db, movie_table, boundaries)
        print('')
        print('Time: {:.1f}s'.format(time.time() - s))

        # Loading the frames for each shot boundary
        # frames = movie_table.load([0], rows=boundaries)
        # montage_img = make_montage(len(boundaries), frames)

        cv2.imwrite('shots.jpg', montage_img)
        print('Successfully generated shots.jpg')

if __name__ == "__main__":
    main()
