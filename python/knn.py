import numpy as np
import sys
import scipy.spatial.distance as dist
from sklearn.neighbors import NearestNeighbors
from decode import load_faster_rcnn_features, load_squeezenet_features, db
from timeit import default_timer
from scanner import JobLoadException
import os
import toml

os.environ['GLOG_minloglevel'] = '4'  # Silencio, Caffe!
import caffe

USE_GPU = True
NET = 'faster_rcnn'
if NET == 'squeezenet':
    FEATURE_LOADER = load_squeezenet_features
    NUM_FEATURES = 1000
elif NET == 'faster_rcnn':
    FEATURE_LOADER = load_faster_rcnn_features
    NUM_FEATURES = 4096
else:
    logging.critical('Unsupported KNN net {}'.format(NET))
    exit()
K = 5

def write(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def write_timer(start):
    write('{:.3f}\n'.format(default_timer() - start))


class FeatureSearch:
    """ TODO(wcrichto): document me """

    def __init__(self, dataset_name, job_name):
        self.index = []
        count = 0
        norms = np.ndarray(shape=(0, NUM_FEATURES))
        write('Loading features... ')
        start = default_timer()
        try:
            results = FEATURE_LOADER(dataset_name, job_name)
        except JobLoadException as err:
            print('Error: either you need to run the knn pipeline with the {} \
net first or you didn\'t update the FEATURE_LOADER.'.format(NET))
            exit()
        for (video, buffers) in results.as_frame_list():
            if NET == 'faster_rcnn':
                cur = count
                for (frame, feats) in buffers:
                    self.index.append(((video, frame), count))
                    count += len(feats)

                norms.resize((count, NUM_FEATURES))
                for (_, feats) in buffers:
                    n = len(feats)
                    if n == 0: continue
                    norms[cur:cur+n,:] = np.array(feats)
                    cur += n
            else:
                for (frame, feats) in buffers:
                    self.index.append(((video, frame), count))
                    norms = np.vstack((norms, feats))
                    count += 1
        write_timer(start)
        write('Preparing KNN... ')
        start = default_timer()
        self.knn = NearestNeighbors(algorithm='brute', n_neighbors=K,
                                    metric='cosine') \
            .fit(norms)
        write_timer(start)

    def search(self, exemplar):
        write('Searching KNN... ')
        start = default_timer()
        _, indices = self.knn.kneighbors(np.array([exemplar]))
        write_timer(start)
        results = []
        for icnt in indices[0]:
            for (j, (vid, jcnt)) in enumerate(self.index):
                if j == len(self.index) - 1 or icnt < self.index[j+1][1]:
                    results.append(vid)
                    break
        assert len(results) == K
        return results


def init_net():
    write('Initializing net... ')
    start = default_timer()

    if USE_GPU: caffe.set_mode_gpu()
    else: caffe.set_mode_cpu()
    scanner_path = db.config.scanner_path
    net_config_path = '{}/features/{}.toml'.format(scanner_path, NET)
    with open(net_config_path, 'r') as f:
        net_config = toml.loads(f.read())

    feature_layer = net_config['net']['output_layers'][-1]

    net = caffe.Net(
        '{}/{}'.format(scanner_path, net_config['net']['model']),
        '{}/{}'.format(scanner_path, net_config['net']['weights']),
        caffe.TEST)

    write_timer(start)

    def process(img_path):
        write('Computing exemplar features... ')
        start = default_timer()
        img = caffe.io.load_image(img_path)
        net.blobs['data'].reshape(*(1, 3, img.shape[0], img.shape[1]))
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        if NET == 'faster_rcnn':
            transformer.set_raw_scale('data', 255)
        else:
            transformer.set_channel_swap('data', (2, 1, 0))

        data = np.asarray([transformer.preprocess('data', img)])
        if NET == 'faster_rcnn':
            net.blobs['data'].data[...] = data
            rois = np.array([[0, 0, img.shape[1], img.shape[0]]])
            net.blobs['rois'].reshape(*(rois.shape))
            net.blobs['rois'].data[...] = rois
            net.forward()
        else:
            net.forward_all(data=data)

        write_timer(start)

        return net.blobs[feature_layer].data.reshape((NUM_FEATURES))

    return process


def main():
    if len(sys.argv) != 3:
        print('Usage: knn.py <job_name> <dataset_name>')
        exit()

    [dataset_name, job_name] = sys.argv[1:]
    searcher = FeatureSearch(dataset_name, job_name)
    get_exemplar_features = init_net()

    while True:
        write('> ')
        img_path = sys.stdin.readline().strip()
        exemplar = get_exemplar_features(img_path)
        print(searcher.search(exemplar))


if __name__ == "__main__":
    main()
