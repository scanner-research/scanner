import math
import os.path

from scannerpy import DeviceType, Job
from scannerpy.stdlib import NetDescriptor
from scannerpy.stdlib.util import temp_directory, download_temp_file
from typing import Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))

import scannerpy
import scannerpy.stdlib.readers as readers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.bboxes as bboxes


@scannerpy.register_python_op()
class BBoxNMS(scannerpy.Kernel):
    def __init__(self, config):
        self.protobufs = config.protobufs
        self.scale = config.args['scale']

    def close(self):
        pass

    def execute(self, *inputs) -> bytes:
        bboxes_list = []
        for c in inputs:
            bboxes_list += readers.bboxes(c, self.protobufs)
        nmsed_bboxes = bboxes.nms(bboxes_list, 0.1)
        return writers.bboxes(nmsed_bboxes, self.protobufs)


def detect_faces(db,
                 input_frame_columns,
                 output_sampler,
                 output_sampler_args,
                 output_names,
                 width=960,
                 prototxt_path=None,
                 model_weights_path=None,
                 templates_path=None,
                 return_profiling=False):
    if prototxt_path is None:
        prototxt_path = download_temp_file(
            'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.prototxt'
        )
    if model_weights_path is None:
        model_weights_path = download_temp_file(
            'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_deploy.caffemodel'
        )
    if templates_path is None:
        templates_path = download_temp_file(
            'https://storage.googleapis.com/scanner-data/nets/caffe_facenet/facenet_templates.bin'
        )

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

    if type(output_names) is not list:
        output_names = [
            '{}_{}'.format(output_names, i)
            for i in range(len(input_frame_columns))
        ]
    else:
        assert (len(output_names) == len(input_frame_columns))

    if type(output_sampler_args) is not list:
        output_sampler_args = [
            output_sampler_args for _ in range(len(input_frame_columns))
        ]
    else:
        assert (len(output_sampler_args) == len(input_frame_columns))

    outputs = []
    scales = [1.0, 0.5, 0.25, 0.125]
    batch_sizes = [int((2**i)) for i in range(len(scales))]
    profilers = {}
    for scale, batch in zip(scales, batch_sizes):
        facenet_args.scale = scale
        caffe_args.batch_size = batch

        frame = db.sources.FrameColumn()
        #resized = db.ops.Resize(
        #    frame = frame,
        #    width = width, height = 0,
        #    min = True, preserve_aspect = True,
        frame_info = db.ops.InfoFromFrame(frame=frame)
        facenet_input = db.ops.FacenetInput(
            frame=frame, args=facenet_args, device=device)
        facenet = db.ops.Facenet(
            facenet_input=facenet_input, args=facenet_args, device=device)
        facenet_output = db.ops.FacenetOutput(
            facenet_output=facenet,
            original_frame_info=frame_info,
            args=facenet_args)
        sampled_output = output_sampler(facenet_output)
        output = db.sinks.Column(columns={'bboxes': sampled_output})

        jobs = []
        for output_name, frame_column, output_sampling in zip(
                output_names, input_frame_columns, output_sampler_args):
            job = Job(
                op_args={
                    frame: frame_column,
                    sampled_output: output_sampling,
                    output: '{}_{}'.format(output_name, scale)
                })
            jobs.append(job)

        output = db.run(
            output,
            jobs,
            force=True,
            work_packet_size=batch * 4,
            io_packet_size=batch * 20,
            pipeline_instances_per_node=pipeline_instances)
        profilers['scale_{}'.format(scale)] = output[0].profiler()
        outputs.append(output)

    # scale = max(width / float(max_width), 1.0)
    scale = 1.0

    bbox_inputs = [db.sources.Column() for _ in outputs]
    nmsed_bboxes = db.ops.BBoxNMS(*bbox_inputs, scale=scale)
    output = db.sinks.Column(columns={'bboxes': nmsed_bboxes})

    jobs = []
    for i in range(len(input_frame_columns)):
        op_args = {}
        for bi, cols in enumerate(outputs):
            op_args[bbox_inputs[bi]] = cols[i].column('bboxes')
        op_args[output] = output_names[i]
        jobs.append(Job(op_args=op_args))

    return db.run(output, jobs, force=True)


def detect_poses(db,
                 input_frame_columns,
                 output_sampler,
                 output_sampler_args,
                 output_name,
                 batch=1,
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

    frame = db.sources.FrameColumn()
    poses_out = db.ops.OpenPose(
        frame=frame, device=device, args=pose_args, batch=batch)
    sampled_poses = output_sampler(poses_out)
    output = db.sinks.Column(columns={'poses': sampled_poses})

    jobs = []
    for i, input_frame_column in enumerate(input_frame_columns):
        job = Job(
            op_args={
                frame: input_frame_column,
                sampled_poses: output_sampler_args,
                output: '{}_{}_poses'.format(output_name, i)
            })
        jobs.append(job)
    output = db.run(
        output,
        jobs,
        force=True,
        work_packet_size=8,
        pipeline_instances_per_node=pipeline_instances)
    return output
