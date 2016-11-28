import numpy as np
import scanner
import toml
import os
import cv2
from timeit import default_timer

os.environ['GLOG_minloglevel'] = '4'  # Silencio, Caffe!
import caffe

USE_GPU = True
NET = 'squeezenet'
BATCH_SIZE = 96
USE_EXPLODED_FRAMES = True
NUM_FRAMES = 139302
db = scanner.Scanner()


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    if USE_GPU: caffe.set_mode_gpu()
    else: caffe.set_mode_cpu()
    scanner_path = db.config.scanner_path
    net_config_path = '{}/features/{}.toml'.format(scanner_path, NET)
    with open(net_config_path, 'r') as f:
        net_config = toml.loads(f.read())

    net = caffe.Net(
        '{}/{}'.format(scanner_path, net_config['net']['model']),
        '{}/{}'.format(scanner_path, net_config['net']['weights']),
        caffe.TEST)

    start = default_timer()
    for _ in range(0, NUM_FRAMES, BATCH_SIZE):
        net.forward()
    print default_timer() - start
    exit()

    width = net_config['net']['input_width']
    height = net_config['net']['input_height']
    transformer = caffe.io.Transformer({'data': (1, 3, height, width)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))

    time_samples = {'io':[], 'preprocess':[], 'net':[]}

    def process_batch(batch_images):
        preprocess_start = default_timer()
        data = np.asarray([transformer.preprocess('data', img) for img in batch_images])
        net.blobs['data'].reshape(*(len(batch_images), 3, height, width))
        preprocess_end = default_timer() - preprocess_start
        time_samples['preprocess'].append(preprocess_end)

        net_start = default_timer()
        net.forward_all(data=data)
        net_end = default_timer() - net_start
        time_samples['net'].append(net_end)

    start = default_timer()

    if USE_EXPLODED_FRAMES:
        with open('meangirls_frames.txt') as f:
            image_paths = [s.strip() for s in f.read().split("\n")][:NUM_FRAMES]

        for batch_paths in chunks(image_paths, BATCH_SIZE):
            io_start = default_timer()
            batch_images = map(caffe.io.load_image, batch_paths)
            io_end = default_timer() - io_start
            time_samples['io'].append(io_end)
            process_batch(batch_images)
    else:
        vid = cv2.VideoCapture('/bigdata/wcrichto/videos/movies/meanGirls.mp4')
        for batch in chunks(range(NUM_FRAMES), BATCH_SIZE):
            io_start = default_timer()
            batch_images = []
            for _ in batch:
                ret, frame = vid.read()
                if ret is False:
                    print 'Error: ran out of video frames'
                    exit()
                batch_images.append(frame)
            io_end = default_timer() - io_start
            time_samples['io'].append(io_end)
            process_batch(batch_images)

    elapsed = default_timer() - start
    print 'Processed {} images (batches of {}) with {} in {:.3f}s @ {:.1f} FPS'.format(
        NUM_FRAMES,
        BATCH_SIZE,
        NET,
        elapsed,
        NUM_FRAMES/elapsed)
    print 'Averages: I/O {:.3f}s, preprocess {:.3f}s, net {:.3f}s'.format(
        np.mean(time_samples['io']),
        np.mean(time_samples['preprocess']),
        np.mean(time_samples['net']))


if __name__ == '__main__':
    main()
