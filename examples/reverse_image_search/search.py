from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import numpy as np
import faiss

db = Database(debug=True)

# with open('rami.jpg') as f:
#     t = f.read()
# db.new_table("query_image", ["img"], [[t]], force=True)
# exit()

descriptor = NetDescriptor.from_file(db, 'nets/yolo.toml')
caffe_args = db.protobufs.CaffeArgs()
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 10

def parse_fvec(bufs, db):
    return np.frombuffer(bufs[0], dtype=np.float32)

def make_op_graph(input):
    caffe_input = db.ops.CaffeInput(
        inputs=[(input, ["frame", "frame_info"])],
        args=caffe_args,
        device=DeviceType.GPU)
    caffe = db.ops.Caffe(
        inputs=[(caffe_input, ["caffe_frame"]), (input, ["frame_info"])],
        args=caffe_args,
        device=DeviceType.GPU)
    # yolo_output = db.ops.YoloOutput(
    #     inputs=[(caffe, ["caffe_output"]), (input, ["frame_info"])],
    #     args=caffe_args)
    return caffe

example = db.table('example')
tasks = db.sampler().range([(example.name(), 'example_yolo')], 0, 100)
# [output_table] = db.run(tasks, make_op_graph(db.ops.Input()), force=True)
output_table = db.table('example_yolo')
# bboxes.draw(example, output_table, 'example_bboxes.mkv')

index = faiss.IndexFlatL2(4096)
for _, vec in output_table.load([0], parse_fvec):
    index.add(np.array([vec]))

table_input = db.ops.Input(["img"])
img_input = db.ops.ImageDecoder(inputs=[(table_input, ["img"])])
[query_output_table] = db.run(db.sampler().all([('query_image', 'query_output')]),
       make_op_graph(img_input),
       force=True)
for _, vec in query_output_table.load([0], parse_fvec):
    print vec.shape
