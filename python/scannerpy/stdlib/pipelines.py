from . import NetDescriptor, writers, bboxes, parsers
from .. import DeviceType, Job
from ..collection import Collection
import math

def detect_faces(db, input_table, sampling, output_name, max_width=960):
    descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    outputs = []
    scales = [1.0, 0.5, 0.25, 0.125]
    batch_sizes = [int((2**i))
                   for i in range(len(scales))]
    for scale, batch in zip(scales, batch_sizes):
        facenet_args.scale = scale
        caffe_args.batch_size = batch

        frame = sampling(input_table.as_op())
        resized = db.ops.Resize(
            frame = frame,
            width = max_width, height = 0,
            min = True, preserve_aspect = True,
            device = DeviceType.GPU)
        frame_info = db.ops.InfoFromFrame(frame = resized)
        facenet_input = db.ops.FacenetInput(
            frame = resized,
            args = facenet_args,
            device = DeviceType.GPU)
        facenet = db.ops.Facenet(
            facenet_input = facenet_input,
            args = facenet_args,
            device = DeviceType.GPU)
        facenet_output = db.ops.FacenetOutput(
            facenet_output = facenet,
            original_frame_info = frame_info,
            args = facenet_args)

        job = Job(
            columns = [facenet_output],
            name = '{}_faces_{}'.format(output_name, scale))
        output = db.run(job, force=True, work_item_size=batch * 4)
        outputs.append(output)

    def make_bbox_table(input, outputs, name):
        all_bboxes = [
            [box for (_, box) in out.load(['bboxes'], parsers.bboxes)]
            for out in outputs]

        nms_bboxes = []
        frames = len(all_bboxes[0])
        runs = len(all_bboxes)

        for fi in range(frames):
            frame_bboxes = []
            for r in range(runs):
                frame_bboxes += (all_bboxes[r][fi])
            frame_bboxes = bboxes.nms(frame_bboxes, 0.3)
            nms_bboxes.append(frame_bboxes)

        _, frame = next(input.load(['frame'], rows=[0]))
        (height, width, _) = frame[0].shape
        scale = max(width / float(max_width), 1.0)
        for bb in nms_bboxes:
            for bbox in bb:
                bbox.x1 *= scale
                bbox.y1 *= scale
                bbox.x2 *= scale
                bbox.y2 *= scale

        return db.new_table(
            name,
            ['bboxes'],
            [[bb] for bb in nms_bboxes],
            writers.bboxes,
            force=True)

    if isinstance(outputs[0], Collection):
        new_tables = []
        for i in range(len(outputs[0].tables())):
            t = make_bbox_table(
                input_table.tables(i),
                [c.tables(i) for c in outputs],
                '{}_bboxes_{}'.format(output_name, i))
            new_tables.append(t)
        return db.new_collection(
            output_name,
            [t.name() for t in new_tables],
            force=True)
    else:
        return make_bbox_table(input_table, outputs, output_name)
