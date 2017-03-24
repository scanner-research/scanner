from scannerpy import Database, DeviceType
from scannerpy.stdlib import NetDescriptor, parsers, bboxes
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
    left_K = np.array([[745.606,0,374.278],
                       [0,746.049,226.198],
                       [0,0,1]],
                      dtype=np.float32)
    left_R = np.array([[0.9680792424,0.02856625033,-0.2490111439],
                       [-0.04880404945,0.9959522125,-0.0754808267],
                       [0.2458469955,0.08522417371,0.965554812]],
                      dtype=np.float64)
    left_t = np.array([[-49.73322909],
                       [142.7355424],
                       [288.2857244]],
                      dtype=np.float32)
    print(left_R.dtype)
    print(left_t)
    print(np.hstack((left_R, left_t)))
    left_P = left_K.dot(np.hstack((left_R, left_t)))
    print(left_P)

    points = np.zeros((1, 3), dtype=np.float32)
    proj_points,_ = cv2.projectPoints(points, left_R, left_t, left_K, None)
    print(proj_points)
    proj_points = left_P.dot(np.array([0, 0, 0, 1]))
    print(proj_points / proj_points[2])

    left_cam, left_rot, left_tr, _, _, _, _ = cv2.decomposeProjectionMatrix(left_P)
    left_t2 = -left_rot.dot(left_tr[:3]/left_tr[3])
    left_tr = left_t2
    proj_points,_ = cv2.projectPoints(points, left_rot, left_tr, left_cam, None)
    print(proj_points)


    left_P2 = left_cam.dot(np.hstack((left_rot, left_tr)))
    print(left_P2)
    proj_points = left_P2.dot(np.array([0, 0, 0, 1]))
    print(proj_points / proj_points[2])

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


    with Database(debug=True) as db:
        left_video = '/n/scanner/apoms/panoptic/160422_mafia2/vgaVideos/vga_01_01.mp4'
        right_video = '/n/scanner/apoms/panoptic/160422_mafia2/vgaVideos/vga_05_01.mp4'
        db.ingest_videos([('left', left_video),
                          ('right', right_video)],
                         force=True)
        left_table = db.table('left')
        right_table = db.table('right')

        gipuma_args = db.protobufs.GipumaArgs()
        gipuma_args.min_disparity = 0
        gipuma_args.max_disparity = 256
        gipuma_args.min_depth = 15
        gipuma_args.max_depth = 300
        gipuma_args.iterations = 16
        gipuma_args.kernel_width = 19
        gipuma_args.kernel_height = 19
        left_camera = gipuma_args.cameras.add()
        for i in range(3):
            for j in range(4):
                left_camera.p.append(left_P[i, j])

        right_camera = gipuma_args.cameras.add()
        for i in range(3):
            for j in range(4):
                right_camera.p.append(right_P[i, j])

        input_op = db.ops.Input(["index", "frame1", "fi1", "frame2", "fi2"])
        op = db.ops.Gipuma(
            inputs=[(input_op, ["frame1", "fi1", "frame2", "fi2"])],
            args=gipuma_args, device=DeviceType.GPU)

        sampler_args = db.protobufs.AllSamplerArgs()
        sampler_args.sample_size = 8
        sampler_args.warmup_size = 0
        tasks = []
        if True:
            task = db.protobufs.Task()
            task.output_table_name = 'left_disparity'
            column_names = [c.name() for c in left_table.columns()]

            sample_left = task.samples.add()
            sample_left.table_name = 'left'
            sample_left.column_names.extend(column_names)
            sample_left.sampling_function = "All"
            sample_left.sampling_args = sampler_args.SerializeToString()

            sample_right = task.samples.add()
            sample_right.table_name = 'right'
            sample_right.column_names.extend(['frame', 'frame_info'])
            sample_right.sampling_function = "All"
            sample_right.sampling_args = sampler_args.SerializeToString()
            tasks.append(task)

        db.run(tasks, op, pipeline_instances_per_node=1, force=True)


if __name__ == "__main__":
    main()
