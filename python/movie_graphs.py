from decode import load_bboxes, load_medians, load_squeezenet_features
from collections import defaultdict
from extract_frames_scanner import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
import tsne
from scipy.spatial import distance

MOVIE = 'anewhope'

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
PERSON = CLASSES.index('person')
SHOT_DIR = '/tmp'
SHOT_TIME_THRESHOLD = 24

def get_shots():
    with open('/bigdata/shots/{}.txt'.format(MOVIE), 'r') as f:
        shots = map(int, f.readlines())
    final_shots = []

    # Filter shots of length less than the threshold
    for i in range(len(shots)-1):
        if shots[i+1] - shots[i] >= SHOT_TIME_THRESHOLD:
            final_shots.append(shots[i])
    return final_shots


def plot_person_shot_histogram():
    shots = get_shots()
    (_, vid_bboxes) = next(load_bboxes(MOVIE, 'features').as_frame_list())

    shot_ppl = defaultdict(list)
    shot_index = 0
    for (frame, frame_bboxes) in vid_bboxes:
        if shot_index < len(shots) - 1 and frame >= shots[shot_index + 1]:
            shot_index += 1
        n = len(filter(lambda x: x.label == PERSON, frame_bboxes))
        shot_ppl[shot_index].append(n)

    totals = []
    for shot, frames in shot_ppl.iteritems():
        frames = np.array(frames)
        totals.append(round(np.mean(frames)))

    hist, bins = np.histogram(totals, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    print bins
    center = (bins[:-1] + bins[1:]) / 2
    plt.clf()
    ax = plt.subplot(1, 1, 1)
    ax.set_title('Mean person count vs. shots for {}'.format(MOVIE))
    ax.set_xlabel('# people detected in shot')
    ax.set_ylabel('# of shots')
    ax.legend()
    ax.bar(center, hist)
    for xy in zip(center, hist):
        ax.annotate(xy[1], xy=xy, textcoords='data')
    plt.savefig('person_shot_histogram_{}.png'.format(MOVIE), dpi=150)


def generate_median_color_montage():
    all_medians = load_medians(MOVIE, 'med').as_frame_list()
    (_, vid_medians) = next(all_medians)
    N = len(vid_medians)
    width = 500
    height = int(math.ceil(float(N) / width))
    out = np.zeros((height, width, 3), np.uint8)
    for (i, (frame, pixels)) in enumerate(vid_medians):
        out[i / width, i % width, :] = pixels
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('median_montage_{}.jpg'.format(MOVIE), out)


# http://stackoverflow.com/questions/17493494/nearest-neighbour-algorithm
def NN(A, start):
    """Nearest neighbor algorithm.
    A is an NxN array indicating distance between N locations
    start is the index of the starting location
    Returns the path and cost of the found solution
    """
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which
    # locations have not been visited
    mask[start] = False
    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]

    return path, cost


def generate_shot_montage(color_sort=False):
    print 'Extracting shots...'
    shots = get_shots()
    N = len(shots)

    print 'Starting montage generation...'
    scale = 0.1
    shot_shape = cv2.imread('{}/0_{:07d}.jpg'.format(SHOT_DIR, shots[0])).shape
    shot_width = int(shot_shape[1] * scale)
    shot_height = int(shot_shape[0] * scale)
    width = 1024
    shots_per_row = width / shot_width
    height = int(shot_height * math.ceil(float(N) / shots_per_row))
    out = np.zeros((height, width, 3), np.uint8)

    if color_sort:
       # http://www.alanzucconi.com/2015/09/30/colour-sorting/
       print 'Finding optimal path...'
       (_, vid_medians) = next(load_medians(MOVIE, 'med').as_frame_list())
       medians = np.array([p for (_, p) in vid_medians])[shots]
       A = distance.squareform(distance.pdist(medians))
       path, _ = NN(A, 0)
    else:
        path = range(N)

    print 'Stitching image...'
    for (i, shot_idx) in enumerate(path):
        shot = shots[shot_idx]
        img = cv2.imread('{}/0_{:07d}.jpg'.format(SHOT_DIR, shot))
        if img is None:
            print 'Bad path'
            exit()
        img = cv2.resize(img, (shot_width, shot_height), interpolation = cv2.INTER_CUBIC)
        row = i / shots_per_row
        col = i % shots_per_row
        out[(row * shot_height):((row+1) * shot_height),
            (col * shot_width):((col+1) * shot_width),
            :] = img

    cv2.imwrite('shot_montage_{}.jpg'.format(MOVIE), out)


def plot_tsne():
    shots = get_shots()
    (_, vid_features) = next(load_squeezenet_features(MOVIE, 'features').as_frame_list())
    features = np.array([p for (_, p) in vid_features])[shots]
    coords = tsne.tsne(features)
    min_x = coords[:, 0].min()
    max_x = coords[:, 0].max()
    min_y = coords[:, 1].min()
    max_y = coords[:, 1].max()

    scale = 1.0
    shot_shape = cv2.imread('{}/0_{:07d}.jpg'.format(SHOT_DIR, shots[0])).shape
    shot_width = int(shot_shape[1] * scale)
    shot_height = int(shot_shape[0] * scale)
    height = 20000
    width = 20000
    out = np.zeros((height, width, 3), np.uint8)

    def feat_coord_to_im_coord(tx, ty):
        ix = int((((tx - min_x) / (max_x - min_x)) * width))
        iy = int((((ty - min_y) / (max_y - min_y)) * height))
        return (ix, iy)

    vid = cv2.VideoWriter(
        'tsne_{}.mp4'.format(MOVIE),
        cv2.VideoWriter_fourcc(*'H264'),
        1.0,
        (width, height))

    for (i, [tx, ty]) in enumerate(coords):
        (ix, iy) = feat_coord_to_im_coord(tx, ty)
        ix -= shot_width/2
        iy -= shot_height/2
        if ix < 0 or ix + shot_width >= width or \
           iy < 0 or iy + shot_height >= height:
            print i, tx, ty
            continue

        img = cv2.imread('{}/0_{:07d}.jpg'.format(SHOT_DIR, shots[i]))
        if img is None:
            print 'Bad path'
            exit()
        img = cv2.resize(img, (shot_width, shot_height), interpolation = cv2.INTER_CUBIC)
        out[iy:(iy+shot_height), ix:(ix+shot_width), :] = img

    for i in range(0, len(coords)-1, 24):
        c1 = feat_coord_to_im_coord(coords[i][0], coords[i][1])
        c2 = feat_coord_to_im_coord(coords[i+1][0], coords[i+1][1])
        cv2.line(out, c1, c2, (0,0,255),5)

    cv2.imwrite('tsne_{}.jpg'.format(MOVIE), out)


def main():
    if False:
        write_indices([('0', i) for i in shots])
        extract_frames({
            'dataset': MOVIE,
            'out_dir': SHOT_DIR
        })

    #plot_person_shot_histogram()
    #generate_median_color_montage()
    #generate_shot_montage(color_sort=False)
    plot_tsne()


if __name__ == "__main__":
    main()
