from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import scipy.misc
import numpy as np
import cv2
import sys
import random

# def parse_fvec(bufs, db):
#     buf = bufs[0]
#     if len(buf) == 1:
#         return []
#     else:
#         splits = len(buf) / (4096*4)
#         return np.split(np.frombuffer(buf, dtype=np.float32), splits)

def main():
    middle_K = np.array([[745.606,0,374.278], [0,746.049,226.198], [0,0,1]],
                      dtype=np.float32)
    middle_R = np.array([[0.9680792424,0.02856625033,-0.2490111439],
                       [-0.04880404945,0.9959522125,-0.0754808267],
                       [0.2458469955,0.08522417371,0.965554812]],
                      dtype=np.float64)
    middle_t = np.array([[-49.73322909],
                       [142.7355424],
                       [288.2857244]],
                      dtype=np.float32)
    middle_P = middle_K.dot(np.hstack((middle_R, middle_t)))

    middle2_K = np.array([
	[745.478,0,368.991],
	[0,745.873,226.521],
	[0,0,1]], dtype=np.float32)
    middle2_R = np.array(
        [
            [0.9776629603,-0.0171559985,-0.2094774638],
            [0.007509628512,0.998877992,-0.04675855584],
            [0.2100446181,0.04414101019,0.9766948498]],
        dtype=np.float64)
    middle2_t = np.array([
	[-64.18295724],
	[141.4067053],
	[284.2669748]],
                         dtype=np.float32)
    middle2_P = middle2_K.dot(np.hstack((middle2_R, middle2_t)))


    right_K = np.array(
        [[744.965,0,379.832],
         [0,745.789,221.489],
         [0,0,1]], dtype=np.float32)
    right_R = np.array([
        [0.5043841257,-0.07024569211,-0.8606173345],
        [-0.1183359922,0.9816581278,-0.1494788048],
        [0.8553322435,0.1772367424,0.4868201828]],
                       dtype=np.float32)
    right_t = np.array([
	[-34.76923864],
	[186.3725964],
	[296.4703366]], dtype=np.float32)
    right_P = right_K.dot(np.hstack((right_R, right_t)))

    left_K = np.array(
	[[742.674,0,373.395],
	 [0,742.973,232.595],
	 [0,0,1]
        ])
    left_R = np.array([
	[0.8303974953,0.1596667216,0.5338038383],
	[-0.1743150237,0.9844146819,-0.02328103501],
	[-0.5292015422,-0.07371751559,0.8452877945]
    ])
    left_t = np.array([
	[-22.6455156],
	[118.6959936],
	[268.0278009]
    ])
    left_P = left_K.dot(np.hstack((left_R, left_t)))

    params = [middle_P, middle2_P, right_P, left_P]


    with Database(debug=False) as db:
        dataset = '160422_haggling1'
        middle_video = '/n/scanner/apoms/panoptic/' + dataset + '/vgaVideos/vga_01_01.mp4'
        middle2_video = '/n/scanner/apoms/panoptic/' + dataset + '/vgaVideos/vga_01_02.mp4'
        right_video = '/n/scanner/apoms/panoptic/' + dataset + '/vgaVideos/vga_05_01.mp4'
        left_video = '/n/scanner/apoms/panoptic/' + dataset + '/vgaVideos/vga_16_13.mp4'
        # db.ingest_videos([('middle', middle_video),
        #                   ('middle2', middle2_video),
        #                   ('right', right_video),
        #                   ('left', left_video)],
        #                  force=True)

        camera_tables = [db.table('middle'),
                          db.table('middle2'),
                          db.table('right'),
                          db.table('left')]

        gipuma_args = db.protobufs.GipumaArgs()
        gipuma_args.min_disparity = 0
        gipuma_args.max_disparity = 384
        gipuma_args.min_depth = 30
        gipuma_args.max_depth = 500
        gipuma_args.iterations = 8
        gipuma_args.kernel_width = 19
        gipuma_args.kernel_height = 19

        for P in params:
            camera = gipuma_args.cameras.add()
            for i in range(3):
                for j in range(4):
                    camera.p.append(P[i, j])

        columns = []
        for i in range(len(params)):
            columns += ["frame" + str(i), "fi" + str(i)]
        input_op = db.ops.Input(["index"] + columns)
        op = db.ops.Gipuma(
            inputs=[(input_op, columns)],
            args=gipuma_args, device=DeviceType.GPU)

        tasks = []

        start = 7000
        end = 7300
        item_size = 64
        sampler_args = db.protobufs.StridedRangeSamplerArgs()
        sampler_args.stride = 1
        while start < end:
            sampler_args.warmup_starts.append(start)
            sampler_args.starts.append(start)
            sampler_args.ends.append(min(start + item_size, end))
            start += item_size

        if True:
            task = db.protobufs.Task()
            task.output_table_name = 'disparity'
            column_names = [c.name() for c in camera_tables[0].columns()]

            sample = task.samples.add()
            sample.table_name = camera_tables[0].name()
            sample.column_names.extend(column_names)
            sample.sampling_function = "StridedRange"
            sample.sampling_args = sampler_args.SerializeToString()

            for t in camera_tables[1:]:
                sample = task.samples.add()
                sample.table_name = t.name()
                sample.column_names.extend(["frame", "frame_info"])
                sample.sampling_function = "StridedRange"
                sample.sampling_args = sampler_args.SerializeToString()

            tasks.append(task)

        output_tables = db.run(tasks, op, pipeline_instances_per_node=4, force=True)
        disparity_table = db.table('disparity')
        for fi, tup in disparity_table.load(['points']):
            points = np.frombuffer(tup[0], dtype=np.float32).reshape(480, 640, 4)
            depth_img = points[:,:,3]
            scipy.misc.toimage(depth_img).save('depth{:05d}.png'.format(fi))
            print(fi)


if __name__ == "__main__":
    main()
