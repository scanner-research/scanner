from . import NetDescriptor, writers, bboxes, poses, parsers
from .. import DeviceType, Job
from ..collection import Collection
import math
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))

def detect_faces(db, input_tables_or_collection, sampling, output_name,
                 width=960):
    descriptor = NetDescriptor.from_file(db, 'nets/caffe_facenet.toml')
    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    if isinstance(input_tables_or_collection, Collection):
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
            #resized = db.ops.Resize(
            #    frame = frame,
            #    width = width, height = 0,
            #    min = True, preserve_aspect = True,
            #    device = DeviceType.GPU)
            frame_info = db.ops.InfoFromFrame(frame = frame)
            facenet_input = db.ops.FacenetInput(
                frame = frame,
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

    # Register nms bbox op and kernel
    db.register_op('BBoxNMS', [], ['poses'], variadic_inputs=True)
    kernel_path = script_dir + '/bbox_nms_kernel.py'
    db.register_python_kernel('BBoxNMS', DeviceType.CPU, kernel_path)
    # scale = max(width / float(max_width), 1.0)
    scale = 1.0
    if isinstance(outputs[0], Collection):
        jobs = []
        for i in range(len(outputs[0].tables())):
            inputs = [c.tables(i) for c in outputs]
            nmsed_bboxes = db.ops.BBoxNMS(*inputs, scale=scale)
            job = Job(
                columns = [nmsed_bboxes],
                name = '{}_boxes_{}'.format(output_name, i))
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
            nmsed_bboxes = db.ops.BBoxNMS(*inputs, scale=scale)
            job = Job(
                columns = [nmsed_bboxes],
                name = '{}_boxes_{}'.format(output_name, i))
            jobs.append(job)
        return db.run(jobs, force=True)


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
    db.register_op('PoseNMS', [], ['poses'], variadic_inputs=True)
    kernel_path = script_dir + '/pose_nms_kernel.py'
    db.register_python_kernel('PoseNMS', DeviceType.CPU, kernel_path)

    if isinstance(outputs[0], Collection):
        jobs = []
        for i in range(len(outputs[0].tables())):
            inputs = [c.tables(i) for c in outputs]
            nmsed_poses = db.ops.PoseNMS(*inputs, height=height)
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
            nmsed_poses = db.ops.PoseNMS(*inputs, height=height)
            job = Job(
                columns = [nmsed_poses],
                name = '{}_poses_{}'.format(output_name, i))
            jobs.append(job)
        return db.run(jobs, force=True)
