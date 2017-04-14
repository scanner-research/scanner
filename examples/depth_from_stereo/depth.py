from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
import scipy.misc
import numpy as np
import cv2
import sys
import random
import json
import time
import os
import os.path
import struct

def write_dmb_file(path, image):
    with open(path, 'wb') as f:
        # type
        f.write(struct.pack('i', 1)) # type
        # height
        f.write(struct.pack('i', image.shape[0]))
        # width
        f.write(struct.pack('i', image.shape[1]))
        # channels
        if len(image.shape) > 2:
            f.write(struct.pack('i', image.shape[2]))
        else:
            f.write(struct.pack('i', 1))
        f.write(image.tobytes())


def make_p_matrices(calib):
    cameras = calib['cameras']
    p_matrices = {}
    for cam in cameras:
        K = np.array(cam['K'])
        R = np.array(cam['R'])
        t = np.array(cam['t'])
        p = K.dot(np.hstack((R, t)))
        p_matrices[(cam['panel'], cam['node'])] = p
    return p_matrices


def main():
    with open('/n/scanner/apoms/panoptic/160422_mafia2/calibration_160422_mafia2.json', 'r') as f:
        calib = json.load(f)

    p_matrices = make_p_matrices(calib)
    dataset = '160422_haggling1'
    template_path = '/n/scanner/apoms/panoptic/' + dataset + '/vgaVideos/vga_{:02d}_{:02d}.mp4'
    i = 0
    video_paths = []
    table_idx = {}
    for p in range(1, 21):
        for c in range(1, 25):
            video_paths.append(template_path.format(p, c))
            table_idx[(p, c)] = i
            i += 1

    with Database(debug=True) as db:
        # Ingest
        if False:
            #collection, _ = db.ingest_video_collection(dataset, video_paths,
            #                                           force=True)
            collection = db.collection(dataset)

            # Setup tables with calibration data
            calibration_table_names = []
            columns = ['P']
            for p in range(1, 21):
                for c in range(1, 25):
                    table_name = 'calibration_{:02d}_{:02d}'.format(p, c)
                    num_rows = collection.tables(len(calibration_table_names)).num_rows()
                    cam = db.protobufs.Camera()
                    if (p == 14 and c == 18) or num_rows == 0:
                        rows = [[cam.SerializeToString()]]
                        db.new_table(table_name, columns, rows, force=True)
                        calibration_table_names.append(table_name)
                        continue
                    P = p_matrices[(p, c)]
                    for i in range(3):
                        for j in range(4):
                            cam.p.append(P[i, j])
                    rows = []
                    for i in range(num_rows):
                        rows.append([cam.SerializeToString()])
                    print(table_name)
                    db.new_table(table_name, columns, rows, force=True)
                    calibration_table_names.append(table_name)
            calib_collection = db.new_collection(dataset + '_calibration',
                                                 calibration_table_names,
                                                 force=True)

        collection = db.collection(dataset)
        calib_collection = db.collection(dataset + '_calibration')

        gipuma_args = db.protobufs.GipumaArgs()
        gipuma_args.min_disparity = 0
        gipuma_args.max_disparity = 384
        gipuma_args.min_depth = 30
        gipuma_args.max_depth = 500
        gipuma_args.iterations = 8
        gipuma_args.kernel_width = 19
        gipuma_args.kernel_height = 19

        columns = []
        camera_groups_length = 4
        for i in range(camera_groups_length):
            columns += ["frame" + str(i), "fi" + str(i), "calib" + str(i)]
        input_op = db.ops.Input(["index"] + columns)
        op = db.ops.Gipuma(
            inputs=[(input_op, columns)],
            args=gipuma_args, device=DeviceType.GPU)

        tasks = []

        start_frame = 4300
        end_frame = 4302
        item_size = 64
        sampler_args = db.protobufs.StridedRangeSamplerArgs()
        sampler_args.stride = 1
        start = start_frame
        end = end_frame
        while start < end:
            sampler_args.warmup_starts.append(start)
            sampler_args.starts.append(start)
            sampler_args.ends.append(min(start + item_size, end))
            start += item_size

        camera_groups = [
            [(1, 1), (1, 2), (5, 1), (16, 13)],
            # [(3, 1), (3, 3), (5, 3), (1, 6)],
            # [(4, 2), (1, 3), (5, 3), (3, 3)],
            # [(7, 4), (7, 8), (6, 3), (8, 3)],
            # [(10, 4), (9, 3), (10, 3), (11, 3)],
            # [(13, 8), (13, 10), (12, 8), (14, 20)],
            # [(16, 4), (16, 16), (15, 2), (16, 8)],
        ]
        for group in camera_groups:
            first_idx = table_idx[group[0]]
            print(first_idx)

            first_table = collection.tables(first_idx)
            first_calib_table = calib_collection.tables(first_idx)

            task = db.protobufs.Task()
            task.output_table_name = 'disparity_{:02d}_{:02d}'.format(
                group[0][0], group[0][1])
            column_names = [c.name() for c in first_table.columns()]

            # Load frames
            sample = task.samples.add()
            sample.table_name = first_table.name()
            sample.column_names.extend(column_names)
            sample.sampling_function = "StridedRange"
            sample.sampling_args = sampler_args.SerializeToString()

            # Load calibration
            sample = task.samples.add()
            sample.table_name = first_calib_table.name()
            sample.column_names.extend(['P'])
            sample.sampling_function = "StridedRange"
            sample.sampling_args = sampler_args.SerializeToString()

            for c, p in group[1:]:
                idx = table_idx[(c, p)]

                print(idx)
                table = collection.tables(idx)
                calib_table = calib_collection.tables(idx)

                sample = task.samples.add()
                sample.table_name = table.name()
                sample.column_names.extend(["frame", "frame_info"])
                sample.sampling_function = "StridedRange"
                sample.sampling_args = sampler_args.SerializeToString()

                sample = task.samples.add()
                sample.table_name = calib_table.name()
                sample.column_names.extend(['P'])
                sample.sampling_function = "StridedRange"
                sample.sampling_args = sampler_args.SerializeToString()

            tasks.append(task)

        # Output data for fusibile
        top_folder = 'gipuma_results/'
        frame_folder = top_folder + '{:08d}/'
        images_folder = frame_folder + 'images/'
        image_name = '{:03d}.png'
        image_path = images_folder + image_name
        krt_path = images_folder + 'cam.txt'
        results_folder = frame_folder + 'results/'
        cam_results_folder = results_folder + '2hat_cam_{:03d}/'
        normals_path = cam_results_folder + 'normals.dmb'
        depth_path = cam_results_folder + 'disp.dmb'

        output_tables = db.run(tasks, op, pipeline_instances_per_node=4, force=True)

        # Export data directory corresponding to image files
        # for i, table in enumerate(collection.tables()):
        #     for fi, tup in table.load(['frame'], rows=range(start_frame,
        #                                                     end_frame)):
        #         if not os.path.exists(images_folder.format(fi)):
        #             os.makedirs(images_folder.format(fi))
        #         img = tup[0]
        #         cv2.imwrite(image_path.format(fi, i), img)
        # Export camera calibration params file (krt_file)
        for fi in range(end_frame - start_frame):
            with open(krt_path.format(fi), 'w') as f:
                f.write(str(479) + '\n')
                i = -1
                offset = 0
                cameras = calib['cameras']
                for p in range(1, 21):
                    for c in range(1, 25):
                        i += 1
                        if p == 14 and c == 18:
                            continue
                        f.write(image_name.format(i) + ' ')
                        cam = cameras[offset]
                        K = cam['K']
                        for n in [item for sublist in K for item in sublist]:
                            f.write(str(n) + ' ')
                        R = cam['R']
                        for n in [item for sublist in R for item in sublist]:
                            f.write(str(n) + ' ')
                        t = cam['t']
                        for n in [item for sublist in t for item in sublist]:
                            f.write(str(n) + ' ')
                        f.write('\n')
                        offset += 1

        # Export normals and depth dmb files
        for i, table in enumerate(output_tables):
            for fi, tup in table.load(['points', 'cost']):
                if not os.path.exists(cam_results_folder.format(fi, i)):
                    os.makedirs(cam_results_folder.format(fi, i))
                points = np.frombuffer(tup[0], dtype=np.float32).reshape(480, 640, 4)
                cost = np.frombuffer(tup[1], dtype=np.float32).reshape(480, 640, 1)
                avg = np.median(cost[:])
                mask = np.where(cost > avg)
                print(len(mask))
                print

                depth_img = points[:,:,3].copy()
                depth_img[mask[0], mask[1]] = 0
                write_dmb_file(depth_path.format(fi, i), depth_img)

                normal_img = points[:,:,0:3].copy()
                normal_img[mask[0],mask[1],:] = 0
                write_dmb_file(normals_path.format(fi, i), normal_img)
                #scipy.misc.toimage(depth_img).save('depth{:05d}_01_01.png'.format(fi))

        # For visualizing depth maps
        if False:
            disparity_table = db.table('disparity_01_01')
            for fi, tup in disparity_table.load(['points']):
                points = np.frombuffer(tup[0], dtype=np.float32).reshape(480, 640, 1)
                avg = np.median(points[:])
                depth_img = points[:,:,0].copy()
                depth_img[np.where(depth_img > avg)] = avg * 10
                print('avg', avg)
                scipy.misc.toimage(depth_img).save('cost{:05d}_01_01.png'.format(fi))

            disparity_table = db.table('disparity_03_01')
            for fi, tup in disparity_table.load(['points']):
                points = np.frombuffer(tup[0], dtype=np.float32).reshape(480, 640, 1)
                depth_img = points[:,:,0]
                scipy.misc.toimage(depth_img).save('cost{:05d}_03_01.png'.format(fi))


if __name__ == "__main__":
    main()
