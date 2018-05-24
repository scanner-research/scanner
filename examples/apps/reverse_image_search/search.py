from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, readers, bboxes
import numpy as np
import faiss
import cv2
import sys
import random

STATIC_DIR = 'examples/reverse_image_search/static'

db = Database(debug=True)

descriptor = NetDescriptor.from_file(db, 'nets/faster_rcnn_coco.toml')
caffe_args = db.protobufs.CaffeArgs()
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 1

def parse_fvec(bufs, db):
    buf = bufs[0]
    if len(buf) == 1:
        return []
    else:
        splits = len(buf) / (4096*4)
        return np.split(np.frombuffer(buf, dtype=np.float32), splits)

def make_op_graph(input):
    caffe_input = db.ops.CaffeInput(
        inputs=[(input, ["frame", "frame_info"])],
        args=caffe_args,
        device=DeviceType.GPU)
    caffe = db.ops.FasterRCNN(
        inputs=[(caffe_input, ["caffe_frame"]), (input, ["frame_info"])],
        args=caffe_args,
        device=DeviceType.GPU)
    frcnn_output = db.ops.FasterRCNNOutput(
        inputs=[(caffe, ["cls_prob", "rois", "fc7"])])
    return frcnn_output

def build_index():
    print('Building index...')
    if not db.has_table('example_frcnn'):
        print('Object detections not found. Running Scanner job...')
        [example], _ = db.ingest_videos(
            [('example', '/bigdata/wcrichto/videos/movies/anewhope.m4v')],
            force=True)
        tasks = db.sampler().strided([(example.name(), 'example_frcnn')], 24)
        db.run(tasks, make_op_graph(db.ops.Input()), force=True)

    output_table = db.table('example_frcnn')
    # bboxes.draw(example, output_table, 'example_bboxes.mkv')

    fvec_index = faiss.IndexFlatL2(4096)
    bbox_index = []
    for (frame, bboxes), (_, vec) in \
        zip(output_table.load([0], readers.bboxes),
            output_table.load([1], parse_fvec)):
        # TODO(wcrichto): fix this frame*24 hack
        if len(vec) > 0:
            fvec_index.add(np.array(vec))
            for bbox in bboxes:
                bbox_index.append((frame*24, bbox))

    return fvec_index, bbox_index

def query(path, fvec_index, bbox_index):
    print('Running query with image {}'.format(path))
    with open(path) as f:
        t = f.read()

    # TODO(wcrichto): fix this silly hack when new_table properly
    # supports force=True
    q_t = "query_image_{}".format(random.randint(0, 1000000))
    db.new_table(q_t, ["img"], [[t]], force=True)

    table_input = db.ops.Input(["img"])
    img_input = db.ops.ImageDecoder(inputs=[(table_input, ["img"])])
    [query_output_table] = db.run(db.sampler().all([(q_t, 'query_output')]),
           make_op_graph(img_input),
           force=True)
    query_output_table = db.table('query_output')
    _, qvecs = next(query_output_table.load([1], parse_fvec))
    if len(qvecs) == 0:
        print('Error: could not find an object in query image.')
        return []

    _, neighbors = fvec_index.search(np.array(qvecs[:1]), 50)
    return [bbox_index[i] for i in neighbors[0]]

def visualize(results):
    example = db.table('example')
    to_vis = []
    for k, (i, bbox) in enumerate(results):
        valid = True
        for j, _1, _2 in to_vis:
            if abs(i - j) < 10:
                valid = False
                break
        if valid:
            to_vis.append((i, bbox, k))
        if len(to_vis) == 5: break

    for i, (frame_index, bbox, k) in enumerate(to_vis):
        _, frame = next(example.load([0], rows=[frame_index]))
        frame = frame[0]
        cv2.rectangle(
            frame,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (255, 0, 0), 3)
        cv2.imwrite('{}/result{}.jpg'.format(STATIC_DIR, i),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '{}/query.jpg'.format(STATIC_DIR)
    fvec_index, bbox_index = build_index()
    results = query(path, fvec_index, bbox_index)
    visualize(results)

if __name__ == "__main__":
    main()
