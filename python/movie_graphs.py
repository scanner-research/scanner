from decode import load_bboxes, load_medians, load_squeezenet_features, db
from collections import defaultdict
from extract_frames_scanner import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import math
import tsne
from scipy.spatial import distance
from pprint import pprint

MOVIE = 'tmp'

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle',
           'airplane', 'bus','train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
           'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase',
           'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier','toothbrush')
PERSON = CLASSES.index('person')

SHOT_DIR = '/tmp'
SHOT_TIME_THRESHOLD = 24

ZOOMS = {
    "CU": 0.5,
    "MS": 0.05,
    "LS": 0
}
ZOOMS_SORT = sorted(ZOOMS, key=ZOOMS.get, reverse=True)

def get_shot_type(areas):
    area = max(areas)
    for zoom in ZOOMS_SORT:
        if area > ZOOMS[zoom]:
            return zoom


def get_shot_people(areas):
    if len(areas) == 1:
        return '1'
    elif len(areas) == 2:
        # if abs(areas[0] - areas[1]) < 0.5:
        #     return '2eq'
        # else:
        #     return '2neq'
        return '2'
    else:
        return 'n'


def get_shots(dataset):
    with open('/bigdata/wcrichto/shots/{}.txt'.format(dataset), 'r') as f:
        shots = map(int, f.readlines())
    final_shots = []

    # Filter shots of length less than the threshold
    for i in range(len(shots)-1):
        if shots[i+1] - shots[i] >= SHOT_TIME_THRESHOLD:
            final_shots.append(shots[i])
    return final_shots


def plot_person_shot_histogram():
    shots = get_shots(MOVIE)
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


def generate_median_color_pixel_montage():
    all_medians = load_medians(MOVIE, 'med').as_frame_list()
    (_, vid_medians) = next(all_medians)
    N = len(vid_medians)
    width = 500
    height = int(math.ceil(float(N) / width))
    out = np.zeros((height, width, 3), np.uint8)
    for (i, (frame, pixels)) in enumerate(vid_medians):
        out[i / width, i % width, :] = pixels
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('median_pixel_montage_{}.png'.format(MOVIE), out)

def generate_median_color_bar_montage():
    all_medians = load_medians(MOVIE, 'med').as_frame_list()
    (_, vid_medians) = next(all_medians)
    step = 480
    N = len(vid_medians)
    width = 720
    bar_height = 4
    height = N/step * bar_height
    out = np.zeros((height, width, 3), np.uint8)
    bar_idx = 0
    for i in range(0, N, step):
        if bar_idx >= N/step: break
        (_, pixel) = vid_medians[i]
        for x in range(width):
            for y in range(bar_height):
                out[bar_idx * bar_height + y, x, :] = pixel
        bar_idx += 1
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite('median_bar_montage_{}.png'.format(MOVIE), out)


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
    shots = get_shots(MOVIE)
    N = len(shots)

    print 'Starting montage generation...'
    #scale = 0.1
    shot_shape = cv2.imread('{}/0_{:07d}.jpg'.format(SHOT_DIR, shots[0])).shape
    #shot_width = int(shot_shape[1] * scale)
    shot_width = 128
    shot_height = int(shot_shape[0] * shot_width/shot_shape[1])
    width = 2048
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

    sort_axis = 'color' if color_sort else 'time'
    cv2.imwrite('shot_montage_{}_by_{}.jpg'.format(MOVIE, sort_axis), out)


def plot_tsne():
    shots = get_shots(MOVIE)
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

def ngram(l, n):
    return zip(*[l[i:] for i in range(n)])

def classify_shots(dataset, stride):
    desc = db.video_descriptors(dataset)['0']
    vid_area = float(desc.width * desc.height)
    (_, vid_bboxes) = next(load_bboxes(dataset, 'patch_features').as_frame_list())
    shots = get_shots(dataset)

    l1 = []
    l2 = []
    for i in range(len(shots)-1):
        (s1, s2) = (shots[i], shots[i+1])
        frame = -1
        for j in range(s1, s2):
            if j % stride == 0:
                frame = j/stride
                break
        if frame == -1: continue
        f, frame_bboxes = vid_bboxes[frame]
        ppl = [p for p in frame_bboxes if p.label == PERSON]
        if len(ppl) == 0: continue
        areas = [(p.x2 - p.x1) * (p.y2 - p.y1) / vid_area for p in ppl]
        ty = get_shot_type(areas)
        ppl = get_shot_people(areas)
        l1.append('{}-{}'.format(ty, ppl))
        l2.append(f)
    return l1, l2

def ngram_histogram():
    stride = 8
    l = classify_shots(MOVIE, stride)

    fig, axarr = plt.subplots(2, 2)
    fig.suptitle("N-gram histograms on shot sequences from {}".format(MOVIE))
    for i in range(0, 4):
        N = i + 2
        ng = ngram(l, N)
        hist = defaultdict(int)
        for k in ng:
            hist[k] += 1

        keys = sorted(hist, key=hist.get, reverse=True)[:10]
        vals = [hist[k] for k in keys]
        for k in keys:
            print k, hist[k]

        ax = axarr[i/2,i%2]
        ax.set_title('{}-grams'.format(N))
        ax.set_ylabel('Frequency')

        x = range(len(keys))
        ax.bar(x, vals, align='edge')
        ax.set_xticks(x)
        ax.set_xticklabels(['/'.join(k) for k in keys], rotation=45)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(6)

        ax.autoscale()

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('ngrams.png'.format(N), dpi=150)


def main():
    # ngram_histogram()
    # exit()

    if True:
        shots = get_shots(MOVIE)
        write_indices([('0', i) for i in shots])
        extract_frames({
            'dataset': MOVIE,
            'out_dir': SHOT_DIR
        })

    #plot_person_shot_histogram()
    generate_median_color_bar_montage()
    generate_shot_montage(color_sort=False)
    generate_shot_montage(color_sort=True)
    #plot_tsne()


if __name__ == "__main__":
    main()
