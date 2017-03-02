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

def main():
    movie_path = util.download_video() if len(sys.argv) <= 1 else sys.argv[1]
    print('Detecting shots in movie {}'.format(movie_path))

    db = Database()
    if not db.has_table('movie'):
        print('Loading movie into Scanner database...')
        db.ingest_videos([('movie', movie_path)], force=True)
    movie_table = db.table('movie')

    if not db.has_table('movie_hist'):
        print('Computing a color histogram for each frame...')
        db.run(
            db.sampler().all([(movie_table.name(), 'movie_hist')]),
            db.ops.Histogram(device=DeviceType.GPU),
            force=True)
    hists_table = db.table('movie_hist')

    print('Computing shot boundaries...')

    # Fetch histograms from disk
    hists = [h for _, h in hists_table.load(['histogram'], parsers.histograms)]

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

    print('Visualizing shot boundaries...')

    # Loading the frames for each shot boundary
    frames = [f[0] for _, f in
              movie_table.load([0], rows=boundaries)]
    n = len(frames)

    (frame_h, frame_w, _) = frames[0].shape
    target_w = 256
    target_h = int(target_w / float(frame_w) * frame_h)
    frames_per_row = 8
    img_w = frames_per_row * target_w
    img_h = int(math.ceil(float(n) / frames_per_row)) * target_h

    img = np.zeros((img_h, img_w, 3))
    for i, frame in enumerate(frames):
        frame = cv2.resize(frame, (target_w, target_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        row = i / frames_per_row
        col = i % frames_per_row
        img[(row * target_h):((row+1) * target_h),
            (col * target_w):((col+1) * target_w),
            :] = frame

    cv2.imwrite('shots.jpg', img)
    print('Successfully generated shots.jpg')

if __name__ == "__main__":
    main()
