import caffe
import numpy as np
import sys
import scipy.spatial.distance as dist
from sklearn.neighbors import NearestNeighbors
from decode import load_features

FEATURE_LAYER = 'fc25'
K = 5

class FeatureSearch:
    def __init__(self, dataset_name, job_name):
        self.index = []
        count = 0
        norms = np.ndarray(shape=(0,4096))
        for video in load_features(dataset_name, job_name):
            self.index.append((video['index'], count))
            bufs = np.array(video['buffers'])
            count += len(bufs)
            norms = np.vstack((norms, np.array(video['buffers'])))
        self.knn = NearestNeighbors(n_neighbors=K).fit(norms)

    def search(self, exemplar):
        _, indices = self.knn.kneighbors(np.array([exemplar]))
        results = []
        for idx in indices[0]:
            for (vid, count) in self.index:
                if idx >= count:
                    results.append((vid, idx - count))
                    break
        assert len(results) == K
        return results


def get_exemplar_features(img_path):
    caffe.set_mode_gpu()

    # TODO: read this from the .toml file
    net = caffe.Net(
        'features/yolo/yolo_deploy.prototxt',
        'features/yolo/yolo.caffemodel',
        caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))

    img = caffe.io.load_image(img_path)
    preprocessed = transformer.preprocess('data', img)

    net.forward_all(data=np.asarray([preprocessed]))
    return net.blobs[FEATURE_LAYER].data[0]


def main():
    [img_path, job_name, dataset_name] = sys.argv[1:]

    exemplar = get_exemplar_features(img_path)
    searcher = FeatureSearch(dataset_name, job_name)
    print searcher.search(exemplar)


if __name__ == "__main__":
    main()
