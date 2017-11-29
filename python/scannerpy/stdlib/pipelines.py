from __future__ import absolute_import, division, print_function, unicode_literals
import math
import os.path

from scannerpy import DeviceType, Job, BulkJob
from scannerpy.stdlib import NetDescriptor, writers, bboxes, poses, parsers
from scannerpy.collection import Collection
from scannerpy.stdlib.util import temp_directory, download_temp_file

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

    if db.has_gpu():
        device = DeviceType.GPU
        pipeline_instances = -1
    else:
        device = DeviceType.CPU
        pipeline_instances = 1

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
        frame_info = db.ops.InfoFromFrame(frame = frame)
        facenet_input = db.ops.FacenetInput(
            frame = frame,
            args = facenet_args,
            device = device)
        facenet = db.ops.Facenet(
            facenet_input = facenet_input,
            args = facenet_args,
            device = device)
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
        output = db.run(bulk_job, force=True, work_packet_size=batch * 4,
                        pipeline_instances_per_node=pipeline_instances)
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


def detect_poses(db, input_frame_columns, sampling, output_name, batch=1,
                 models_path=None,
                 pose_model_weights_path=None,
                 hand_prototxt_path=None,
                 hand_model_weights_path=None,
                 face_prototxt_path=None,
                 face_model_weights_path=None):
    if models_path is None:
        models_path = os.path.join(temp_directory(), 'openpose')

        pose_fs_url = 'http://posefs1.perception.cs.cmu.edu/OpenPose/models/'
        # Pose prototxt
        download_temp_file(
            'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
            'openpose/master/models/pose/coco/pose_deploy_linevec.prototxt',
            'openpose/pose/coco/pose_deploy_linevec.prototxt')
        # Pose model weights
        download_temp_file(
            os.path.join(pose_fs_url, 'pose/coco/pose_iter_440000.caffemodel'),
            'openpose/pose/coco/pose_iter_440000.caffemodel')
        # Hands prototxt
        download_temp_file(
            'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
            'openpose/master/models/hand/pose_deploy.prototxt',
            'openpose/hand/pose_deploy.prototxt')
        # Hands model weights
        download_temp_file(
            os.path.join(pose_fs_url, 'hand/pose_iter_102000.caffemodel'),
            'openpose/hand/pose_iter_102000.caffemodel')
        # Face prototxt
        download_temp_file(
            'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
            'openpose/master/models/face/pose_deploy.prototxt',
            'openpose/face/pose_deploy.prototxt')
        # Face model weights
        download_temp_file(
            os.path.join(pose_fs_url, 'face/pose_iter_116000.caffemodel'),
            'openpose/face/pose_iter_116000.caffemodel')
        # Face haar cascades
        download_temp_file(
            'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/'
            'openpose/master/models/face/haarcascade_frontalface_alt.xml',
            'openpose/face/haarcascade_frontalface_alt.xml')

    pose_args = db.protobufs.OpenPoseArgs()
    pose_args.model_directory = models_path
    pose_args.pose_num_scales = 3
    pose_args.pose_scale_gap = 0.33
    pose_args.hand_num_scales = 4
    pose_args.hand_scale_gap = 0.4

    if db.has_gpu():
        device = DeviceType.GPU
        pipeline_instances = -1
    else:
        device = DeviceType.CPU
        pipeline_instances = 1

    frame = db.ops.FrameInput()
    poses_out = db.ops.OpenPose(
        frame=frame,
        device=device,
        args=pose_args,
        batch=batch)
    sampled_poses = poses_out.sample()
    output = db.ops.Output(columns=[sampled_poses])

    jobs = []
    for i, input_frame_column in enumerate(input_frame_columns):
        job = Job(op_args={
            frame: input_frame_column,
            sampled_poses: sampling,
            output: '{}_{}_poses'.format(output_name, i)
        })
        jobs.append(job)
    bulk_job = BulkJob(output=output, jobs=jobs)
    output = db.run(bulk_job, force=True, work_packet_size=8,
                    pipeline_instances_per_node=pipeline_instances)
    return output
