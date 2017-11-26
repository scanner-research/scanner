from . import NetDescriptor, writers, bboxes, poses, parsers
from .. import DeviceType, Job, BulkJob
from ..collection import Collection
from .util import download_temp_file
import math
import os.path

script_dir = os.path.dirname(os.path.abspath(__file__))

def detect_faces(db, input_frame_columns, output_sampling, output_name,
                 width=960, prototxt_path=None, model_weights_path=None,
                 templates_path=None,
                 return_profiling=False):
    if prototxt_path is None:
        prototxt_path = download_temp_file(
            'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.prototxt')
    if model_weights_path is None:
        model_weights_path = download_temp_file(
            'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.caffemodel')
    if templates_path is None:
        templates_path = download_temp_file(
            'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_templates.bin')

    descriptor = NetDescriptor(db)
    descriptor.model_path = prototxt_path
    descriptor.model_weights_path = model_weights_path
    descriptor.input_layer_names = ['data']
    descriptor.output_layer_names = ['score_final']
    descriptor.mean_colors = [119.29959869, 110.54627228, 101.8384321]

    facenet_args = db.protobufs.FacenetArgs()
    facenet_args.templates_path = templates_path
    facenet_args.threshold = 0.5
    caffe_args = facenet_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())

    outputs = []
    scales = [1.0, 0.5, 0.25, 0.125]
    batch_sizes = [int((2**i))
                   for i in range(len(scales))]
    profilers = {}
    for scale, batch in zip(scales, batch_sizes):
        facenet_args.scale = scale
        caffe_args.batch_size = batch

        frame = db.ops.FrameInput()
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
        sampled_output = facenet_output.sample()
        output = db.ops.Output(columns=[sampled_output])

        jobs = []
        for i, frame_column in enumerate(input_frame_columns):
            job = Job(op_args={
                frame: frame_column,
                sampled_output: output_sampling,
                output: '{}_{}_faces_{}'.format(output_name, i, scale)
            })
            jobs.append(job)

        bulk_job = BulkJob(output=output, jobs=jobs)
        output = db.run(bulk_job, force=True, work_packet_size=batch * 4)
        profilers['scale_{}'.format(scale)] = output[0].profiler()
        outputs.append(output)

    # Register nms bbox op and kernel
    db.register_op('BBoxNMS', [], ['bboxes'], variadic_inputs=True)
    kernel_path = script_dir + '/bbox_nms_kernel.py'
    db.register_python_kernel('BBoxNMS', DeviceType.CPU, kernel_path)
    # scale = max(width / float(max_width), 1.0)
    scale = 1.0

    bbox_inputs = [db.ops.Input() for _ in outputs]
    nmsed_bboxes = db.ops.BBoxNMS(*bbox_inputs, scale=scale)
    output = db.ops.Output(columns=[nmsed_bboxes])

    jobs = []
    for i in range(len(input_frame_columns)):
        op_args = {}
        for bi, cols in enumerate(outputs):
            op_args[bbox_inputs[bi]] = cols[i].column('bboxes')
        op_args[output] = '{}_boxes_{}'.format(output_name, i)
        jobs.append(Job(op_args=op_args))
    bulk_job = BulkJob(output=output, jobs=jobs)
    return db.run(bulk_job, force=True)


def detect_poses(db, input_frame_columns, sampling, output_name, height=480):
    descriptor = NetDescriptor.from_file(db, 'nets/cpm2.toml')
    cpm2_args = db.protobufs.CPM2Args()
    caffe_args = cpm2_args.caffe_args
    caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
    caffe_args.batch_size = 1

    outputs = []
    scales = [1.0, 0.7, 0.49, 0.343]
    for scale in scales:
        cpm2_args.scale = 368.0/height * scale

        frame = db.ops.FrameInput()
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
        sampled_poses = poses_out.sample()
        output = db.ops.Output(columns=[sampled_poses])

        jobs = []
        for i, input_frame_column in enumerate(input_frame_columns):
            job = Job(op_args={
                frame: input_frame_column,
                output: '{}_{}_poses_{}'.format(output_name, i, scale)
            })
            jobs.append(job)
        bulk_job = BulkJob(output=output, jobs=jobs)
        output = db.run(bulk_job, force=True, work_packet_size=8)
        outputs.append(output)

    # Register nms pose op and kernel
    db.register_op('PoseNMS', [], ['poses'], variadic_inputs=True)
    kernel_path = script_dir + '/pose_nms_kernel.py'
    db.register_python_kernel('PoseNMS', DeviceType.CPU, kernel_path)

    pose_inputs = [db.ops.Input() for _ in outputs]
    nmsed_poses = db.ops.PoseNMS(*bbox_inputs, height=height)
    output = db.ops.Output(columns=[nmsed_poses])

    jobs = []
    for i in range(len(input_frame_columns)):
        op_args = {}
        for bi, cols in enumerate(outputs):
            op_args[pose_inputs[bi]] = cols[i]
        op_args[output] = '{}_poses_{}'.format(output_name, i)
        job = Job(op_args=op_args)
        jobs.append(job)
    bulk_job = BulkJob(output=output, jobs=jobs)
    outputs =  db.run(bulk_job, force=True)
    profilers['nms'] = outputs[0].profiler()
    return outputs
