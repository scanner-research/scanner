from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import readers
from scipy.spatial import distance
from subprocess import check_call as run
import numpy as np
import cv2
import math
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
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
    #data = [go.Scatter(x=range(n),y=diffs)]
    #offline.plot(data, filename='histogram-diffs.html')

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

def main(movie_path):
    total_start = time.time()

    print('Detecting shots in movie {}'.format(movie_path))
    movie_name = os.path.basename(movie_path)

    # Use GPU kernels if we have a GPU
    db = Database()

    print('Loading movie into Scanner database...')
    s = time.time()

    if db.has_gpu():
        device = DeviceType.GPU
    else:
        device = DeviceType.CPU

    ############ ############ ############ ############
    # 0. Ingest the video into the database
    ############ ############ ############ ############
    [movie_table], _ = db.ingest_videos([(movie_name, movie_path)],
                                        force=True)
    print('Time: {:.1f}s'.format(time.time() - s))
    print('Number of frames in movie: {:d}'.format(movie_table.num_rows()))

    s = time.time()
    ############ ############ ############ ############
    # 1. Run Histogram over the entire video in Scanner
    ############ ############ ############ ############
    print('Computing a color histogram for each frame...')
    frame = db.sources.FrameColumn()
    histogram = db.ops.Histogram(
        frame = frame,
        device = device)
    output = db.sinks.Column(columns={'histogram': histogram})
    job = Job(op_args={
        frame: movie_table.column('frame'),
        output: movie_name + '_hist'
    })
    [hists_table] = db.run(output=output, jobs=[job], force=True)
    print('\nTime: {:.1f}s, {:.1f} fps'.format(
        time.time() - s,
        movie_table.num_rows() / (time.time() - s)))

    s = time.time()
    ############ ############ ############ ############
    # 2. Load histograms and compute shot boundaries
    #    in python
    ############ ############ ############ ############
    print('Computing shot boundaries...')
    # Read histograms from disk
    hists = [h for h in
             hists_table.column('histogram').load(readers.histograms)]
    boundaries = compute_shot_boundaries(hists)
    print('Found {:d} shots.'.format(len(boundaries)))
    print('Time: {:.1f}s'.format(time.time() - s))

    s = time.time()
    ############ ############ ############ ############
    # 3. Create montage in Scanner
    ############ ############ ############ ############
    print('Creating shot montage...')

    row_length = min(16, len(boundaries))
    rows_per_item = 1
    target_width = 256

    # Compute partial row montages that we will stack together
    # at the end
    item_size = row_length * rows_per_item

    starts_remainder = len(boundaries) % item_size
    evenly_divisible = (starts_remainder == 0)
    if not evenly_divisible:
        boundaries = boundaries[0:len(boundaries) - starts_remainder]

    frame = db.sources.FrameColumn()
    gather_frame = db.streams.Gather(frame, boundaries)
    sliced_frame = db.streams.Slice(
        gather_frame, partitioner=db.partitioner.all(item_size))
    montage = db.ops.Montage(
        frame = sliced_frame,
        num_frames = row_length * rows_per_item,
        target_width = target_width,
        frames_per_row = row_length,
        device = device)
    sampled_montage = db.streams.Gather(montage)
    output = db.sinks.Column(
        columns={'montage': db.streams.Unslice(sampled_montage).lossless()})

    job = Job(op_args={
        frame: movie_table.column('frame'),
        sampled_montage: [[item_size - 1]
                          for _ in range(len(boundaries) // item_size)],
        output: 'montage_image'
    })
    [montage_table] = db.run(output=output, jobs=[job], force=True)

    # Stack all partial montages together
    montage_img = np.zeros((1, target_width * row_length, 3), dtype=np.uint8)
    for img in montage_table.column('montage').load():
        img = np.flip(img, 2)
        montage_img = np.vstack((montage_img, img))

    print('')
    print('Time: {:.1f}s'.format(time.time() - s))

    ############ ############ ############ ############
    # 4. Write montage to disk
    ############ ############ ############ ############
    cv2.imwrite('shots.jpg', montage_img)
    print('Successfully generated shots.jpg')
    print('Total time: {:.2f} s'.format(time.time() - total_start))


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print('Usage: main.py <video_file>')
        exit(1)
    main(sys.argv[1])
