import numpy as np
import sys
import scipy.spatial.distance as dist
from sklearn.neighbors import NearestNeighbors
from decode import load_squeezenet_features, db
from timeit import default_timer
import os
import toml

os.environ['GLOG_minloglevel'] = '4' # Silencio, Caffe!
import caffe

USE_GPU = False
NET = 'squeezenet'
NUM_FEATURES = 1000
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
        norms = np.ndarray(shape=(0,NUM_FEATURES))
        write('Loading features... ')
        start = default_timer()
        for video in load_squeezenet_features(dataset_name, job_name):
            self.index.append((video['path'], count))
            bufs = np.array(video['buffers'])
            count += len(bufs)
            norms = np.vstack((norms, np.array(video['buffers'])))
        write_timer(start)
        write('Preparing KNN... ')
        start = default_timer()
        self.knn = NearestNeighbors(algorithm='brute', n_neighbors=K,
                                    metric='euclidean') \
            .fit(norms)
        write_timer(start)


    def search(self, exemplar):
        write('Searching KNN... ')
        start = default_timer()
        _, indices = self.knn.kneighbors(np.array([exemplar]))
        write_timer(start)
        results = []
        for idx in indices[0]:
            for (vid, count) in self.index:
                if idx >= count:
                    results.append((vid, idx - count))
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

    feature_layer = net_config['net']['output_layers'][0]

    net = caffe.Net(
        '{}/{}'.format(scanner_path, net_config['net']['model']),
        '{}/{}'.format(scanner_path, net_config['net']['weights']),
        caffe.TEST)

    write_timer(start)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))

    def process(img_path):
        write('Computing exemplar features... ')
        start = default_timer()
        img = caffe.io.load_image(img_path)
        preprocessed = transformer.preprocess('data', img)
        net.forward_all(data=np.asarray([preprocessed]))
        write_timer(start)

        return net.blobs[feature_layer].data[0].reshape((NUM_FEATURES))

    return process


def main():
    [job_name, dataset_name] = sys.argv[1:]
    searcher = FeatureSearch(dataset_name, job_name)
    get_exemplar_features = init_net()

    while True:
        sys.stdout.write('> ')
        img_path = sys.stdin.readline().strip()
        exemplar = get_exemplar_features(img_path)
        print searcher.search(exemplar)


if __name__ == "__main__":
    main()
