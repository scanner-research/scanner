import scannerpy
from scannerpy import Database, Job, DeviceType, FrameType
from typing import Sequence
import cv2
import numpy as np

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial provides an overview of the more advanced features of Scanner  #
# ops and when you would want to use them.                                     #
################################################################################

db = Database()

example_video_path = util.download_video()
[input_table], _ = db.ingest_videos([('example', example_video_path)],
                                    force=True)

videos = []

# Many ops simply involve applying some processing to their inputs and then
# returning their outputs. But there are also many operations in video
# processing that require the ability to see adjacent frames (such as for
# computing optical flow), need to keep state over time (such as for tracking
# objects), or need to process multiple elements for efficiency reasons (such as
# batching for DNNs).

# Scanner ops therefore have several optional attributes that enable them to
# support these forms of operations:


# 1. Device Type:
#   Ops can specify that they require CPUs or GPUs by declaring their device
#   type. By default, the device_type is DeviceType.CPU.

@scannerpy.register_python_op(device_type=DeviceType.CPU)
def device_resize(config, frame: FrameType) -> FrameType:
    return cv2.resize(frame, (config.args['width'], config.args['height']))

frame = db.sources.FrameColumn()
resized_frame = db.ops.device_resize(frame=frame, width=640, height=480)
output = db.sinks.FrameColumn(columns={'frame': resized_frame})

job = Job(op_args={
    frame: input_table.column('frame'),
    output: 'example_resize'
})

[table] = db.run(output=output, jobs=[job], force=True)

table.column('frame').save_mp4('02_device_resize')

videos.append('02_device_resize.mp4')

# 2. Batch:
#   The Op can receive multiple elements at once to enable SIMD or
#   vector-style processing.

@scannerpy.register_python_op(batch=10)
def batch_resize(config, frame: Sequence[FrameType]) -> Sequence[FrameType]:
    output_frames = []
    for fr in frame:
        output_frames.append(cv2.resize(
            fr, (config.args['width'], config.args['height'])))
    return output_frames

# Here we specify that the resize op should receive a batch of 10
# input elements at once. Logically, each element is still processed
# independently but multiple elements are provided to enable efficient
# batch processing. If there are not enough elements left in a stream,
# the Op may receive less than a batch worth of elements.

frame = db.sources.FrameColumn()
resized_frame = db.ops.batch_resize(frame=frame, width=640, height=480,
                                    batch=10)
output = db.sinks.FrameColumn(columns={'frame': resized_frame})

job = Job(op_args={
    frame: input_table.column('frame'),
    output: 'example_batch_resize'
})

[table] = db.run(output=output, jobs=[job], force=True)

table.column('frame').save_mp4('02_batch_resize')

videos.append('02_batch_resize.mp4')


# 3. Stencil:
#   The Op requires a window of input elements (for example, the
#   previous and next element) at the same time to produce an
#   output.

# Here, we use the stencil attribute to write an optical flow op which
# computes flow between the current and next frame.
@scannerpy.register_python_op(stencil=[0, 1])
def optical_flow(config, frame: Sequence[FrameType]) -> FrameType:
    gray1 = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5,
                                        1.2, 0)
    return flow


# This op visualizes the flow field by converting it into an rgb image
@scannerpy.register_python_op()
def visualize_flow(config, flow: FrameType) -> FrameType:
    hsv = np.zeros(shape=(flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

frame = db.sources.FrameColumn()
# This next line is using a feature we'll discuss in the next tutorial, but you
# can think of it as selecting a subset of elements from the stream (here,
# frames 0 to 30)
range_frame = db.streams.Range(frame, 0, 30)
flow = db.ops.optical_flow(frame=range_frame, stencil=[0, 1])
flow_viz = db.ops.visualize_flow(flow=flow)
output = db.sinks.FrameColumn(columns={'flow_viz': flow_viz})

job = Job(op_args={
    frame: input_table.column('frame'),
    output: 'example_flow'
})

[table] = db.run(output=output, jobs=[job], force=True)

table.column('flow_viz').save_mp4('02_flow')

videos.append('02_flow.mp4')

# 4. Bounded State:
#   For each output, the Op requires at least W sequential
#   "warmup" elements before it can produce a valid output.
#   For example, if the output of this Op is sampled
#   sparsely, this guarantees that the Op can "warmup"
#   its state on a stream of W elements before producing the
#   requested output.

import subprocess

@scannerpy.register_python_op(bounded_state=60)
class BackgroundSubtraction(scannerpy.Kernel):
    def __init__(self, config):
        self.config = config
        self.alpha = config.args['alpha']
        self.thresh = config.args['threshold']

    # Reset is called when the kernel switches to a new part of the stream
    # and so shouldn't maintain it's previous state
    def reset(self):
        self.average_image = None

    def execute(self, frame: FrameType) -> FrameType:
        if self.average_image is None:
            self.average_image = frame

        mask = np.abs(frame - self.average_image) < 255 * self.thresh
        mask = np.any(mask, axis=2)

        masked_image = np.copy(frame)
        wmask = np.where(mask)
        masked_image[wmask[0], wmask[1], :] = 0

        self.average_image = (self.average_image * (1.0 - self.alpha) +
                              frame * self.alpha)

        return masked_image

# Here we wrote an op that performs background subtraction by keeping a
# running average image over the past frames. We set `bounded_state=60`
# to indicate that this kernel needs at least 60 frames before the output
# should be considered reasonable.

# First, we download a static camera video from youtube
# subprocess.check_call(
#     'youtube-dl -f 137 \'https://youtu.be/cVHqFqNz7eM\' -o test.mp4',
#     shell=True)
# [static_table], _ = db.ingest_videos([('static_video', 'test.mp4')],
#                                     force=True)
static_table = input_table

# Then we perform background subtraction and indicate we need 60 prior
# frames to produce correct output
frame = db.sources.FrameColumn()
masked_frame = db.ops.BackgroundSubtraction(frame=frame,
                                            alpha=0.05, threshold=0.05,
                                            bounded_state=60)
# Here, we say that we only want the outputs for this range of frames
sampled_frame = db.streams.Range(masked_frame, 0, 120)
output = db.sinks.Column(columns={'frame': sampled_frame})
job = Job(op_args={
    frame: static_table.column('frame'),
    output: 'masked_video',
})
[table] = db.run(output=output, jobs=[job], force=True)

table.column('frame').save_mp4('02_masked')

videos.append('02_masked.mp4')


# 5. Unbounded State:
#     This Op will always process all preceding elements of
#     its input streams before producing a requested output.
#     This means that sampling operations after this Op
#     can not change how many inputs it receives. In the next
#     tutorial, we will show how this can be relaxed for
#     sub-streams of the input.
@scannerpy.register_python_op(unbounded_state=True)
class Example(scannerpy.Kernel):
    def __init__(self, config):
        pass

    def reset(self):
        pass

    def execute(self, frame: FrameType) -> bytes:
        pass


print('Finished! The following videos were written: {:s}'
      .format(', '.join(videos)))
