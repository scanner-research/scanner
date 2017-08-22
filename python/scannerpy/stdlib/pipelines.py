from . import NetDescriptor, writers, bboxes, poses, parsers
from .. import DeviceType, Job
from ..collection import Collection
import math
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))

def detect_faces(db, input_tables_or_collection, sampling, output_name,
                 max_width=960):
    descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    if isinstance(outputs[0], Collection):
        input_tables = [input_tables_or_collection]
    else:
        input_tables = input_tables_or_collection

    outputs = []
    scales = [1.0, 0.5, 0.25, 0.125]
    batch_sizes = [int((2**i))
                   for i in range(len(scales))]
    for scale, batch in zip(scales, batch_sizes):
        facenet_args.scale = scale
        caffe_args.batch_size = batch

        jobs = []
        for input_table in input_tables:
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
            jobs.append(job)
        output = db.run(jobs, force=True, work_item_size=batch * 4)
        outputs.append(output)

    def make_bbox_table(input_table, outputs, name):
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
            frame_bboxes = bboxes.nms(frame_bboxes, 0.1)
            nms_bboxes.append(frame_bboxes)

        _, frame = next(input_table.load(['frame'], rows=[0]))
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
                input_tables.tables(i),
                [c.tables(i) for c in outputs],
                '{}_bboxes_{}'.format(output_name, i))
            new_tables.append(t)
        return db.new_collection(
            output_name,
            [t.name() for t in new_tables],
            force=True)
    else:
        new_tables = []
        for i in range(len(outputs)):
            t = make_bbox_table(
                input_tables[i],
                [c[i] for c in outputs],
                '{}_bboxes_{}'.format(output_name, i))
            new_tables.append(t)
        return new_tables


def detect_poses(db, input_tables_or_collection, sampling, output_name,
                 height=480):
    descriptor = NetDescriptor.from_file(db, 'nets/cpm2.toml')
    cpm2_args = db.protobufs.CPM2Args()
    caffe_args = cpm2_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 1

    if isinstance(input_tables_or_collection, Collection):
        input_tables = [input_tables_or_collection]
    else:
        input_tables = input_tables_or_collection

    outputs = []
    scales = [1.0, 0.7, 0.49, 0.343]
    for scale in scales:
        cpm2_args.scale = 368.0/height * scale
        jobs = []
        for input_table in input_tables:
            frame = sampling(input_table.as_op())
            frame_info = db.ops.InfoFromFrame(frame = frame)
            cpm2_input = db.ops.CPM2Input(
                frame = frame,
                args = cpm2_args,
                device = DeviceType.GPU)
            cpm2_resized_map, cpm2_joints = db.ops.CPM2(
                cpm2_input = cpm2_input,
                args = cpm2_args,
                device = DeviceType.GPU)
            poses_out = db.ops.CPM2Output(
                cpm2_resized_map = cpm2_resized_map,
                cpm2_joints = cpm2_joints,
                original_frame_info = frame_info,
                args = cpm2_args)
            job = Job(
                columns = [poses_out],
                name = '{}_poses_{}'.format(output_name, scale))
            jobs.append(job)
        output = db.run(jobs, force=True, work_item_size=8)
        outputs.append(output)

    # Register nms pose op and kernel
    db.register_op('PoseNMSKernel', [], ['poses'], variadic_inputs=True)
    kernel_path = script_dir + '/pose_nms_kernel.py'
    db.register_python_kernel('PoseNMSKernel', DeviceType.CPU, kernel_path)

    if isinstance(outputs[0], Collection):
        jobs = []
        for i in range(len(outputs[0].tables())):
            inputs = [c.tables(i) for c in outputs]
            nmsed_poses = db.ops.PoseNMSKernel(*inputs, height=height)
            job = Job(
                columns = [nmsed_poses],
                name = '{}_poses_{}'.format(output_name, i))
            jobs.append(job)
        out_tables = db.run(jobs, force=True)
        return db.new_collection(
            output_name,
            [t.name() for t in out_tables],
            force=True)
    else:
        jobs = []
        for i in range(len(outputs[0])):
            inputs = [c[i].as_op().all() for c in outputs]
            nmsed_poses = db.ops.PoseNMSKernel(*inputs, height=height)
            job = Job(
                columns = [nmsed_poses],
                name = '{}_poses_{}'.format(output_name, i))
            jobs.append(job)
        return db.run(jobs, force=True)
