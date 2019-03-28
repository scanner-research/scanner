import scannerpy as sp
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

def main():
    videos = []
    # Keep track of the streams so we can delete them at the end
    streams = []

    cl = sp.Client()

    example_video_path = util.download_video()
    video_stream = sp.NamedVideoStream(cl, 'example', path=example_video_path)
    streams.append(video_stream)
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

    @sp.register_python_op(device_type=sp.DeviceType.CPU)
    def device_resize(config, frame: sp.FrameType) -> sp.FrameType:
        return cv2.resize(frame, (config.args['width'], config.args['height']))

    frames = cl.io.Input([video_stream])

    resized_frames = cl.ops.device_resize(frame=frames, width=640, height=480)

    stream = sp.NamedVideoStream(cl, 'example_resize')
    streams.append(stream)
    output = cl.io.Output(resized_frames, [stream])

    cl.run(output, sp.PerfParams.estimate())

    stream.save_mp4('02_device_resize')
    videos.append('02_device_resize.mp4')

    # 2. Batch:
    #   The Op can receive multiple elements at once to enable SIMD or
    #   vector-style processing.

    @sp.register_python_op(batch=10)
    def batch_resize(config, frame: Sequence[sp.FrameType]) -> Sequence[sp.FrameType]:
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

    frame = cl.io.Input([video_stream])

    resized_frame = cl.ops.batch_resize(frame=frame, width=640, height=480,
                                        batch=10)

    stream = sp.NamedVideoStream(cl, 'example_batch_resize')
    streams.append(stream)
    output = cl.io.Output(resized_frame, [stream])

    cl.run(output, sp.PerfParams.estimate())

    stream.save_mp4('02_batch_resize')
    videos.append('02_batch_resize.mp4')


    # 3. Stencil:
    #   The Op requires a window of input elements (for example, the
    #   previous and next element) at the same time to produce an
    #   output.

    # Here, we use the stencil attribute to write an optical flow op which
    # computes flow between the current and next frame.
    @sp.register_python_op(stencil=[0, 1])
    def optical_flow(config, frame: Sequence[sp.FrameType]) -> sp.FrameType:
        gray1 = cv2.cvtColor(frame[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5,
                                            1.2, 0)
        return flow


    # This op visualizes the flow field by converting it into an rgb image
    @sp.register_python_op()
    def visualize_flow(config, flow: sp.FrameType) -> sp.FrameType:
        hsv = np.zeros(shape=(flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb

    frames = cl.io.Input([video_stream])

    # This next line is using a feature we'll discuss in the next tutorial, but you
    # can think of it as selecting a subset of elements from the stream (here,
    # frames 0 to 30)
    range_frames = cl.streams.Range(frames, [(0, 30)])
    flows = cl.ops.optical_flow(frame=range_frames, stencil=[0, 1])
    flow_viz_frames = cl.ops.visualize_flow(flow=flows)

    stream = sp.NamedVideoStream(cl, 'example_flow')
    streams.append(stream)
    output = cl.io.Output(flow_viz_frames, [stream])

    cl.run(output, sp.PerfParams.estimate())

    stream.save_mp4('02_flow')
    videos.append('02_flow.mp4')

    # 4. Bounded State:
    #   For each output, the Op requires at least W sequential
    #   "warmup" elements before it can produce a valid output.
    #   For example, if the output of this Op is sampled
    #   sparsely, this guarantees that the Op can "warmup"
    #   its state on a stream of W elements before producing the
    #   requested output.

    import subprocess

    @sp.register_python_op(bounded_state=60)
    class BackgroundSubtraction(sp.Kernel):
        def __init__(self, config, alpha, threshold):
            self.config = config
            self.alpha = alpha
            self.thresh = threshold

        # Reset is called when the kernel switches to a new part of the stream
        # and so shouldn't maintain it's previous state
        def reset(self):
            self.average_image = None

        def execute(self, frame: sp.FrameType) -> sp.FrameType:
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

    frames = cl.io.Input([video_stream])

    # We perform background subtraction and indicate we need 60 prior
    # frames to produce correct output
    masked_frames = cl.ops.BackgroundSubtraction(frame=frames,
                                                 alpha=0.05, threshold=0.05,
                                                 bounded_state=60)
    # Here, we say that we only want the outputs for this range of frames
    sampled_frames = cl.streams.Range(masked_frames, [(0, 120)])

    stream = sp.NamedVideoStream(cl, 'masked_video')
    streams.append(stream)
    output = cl.io.Output(sampled_frames, [stream])

    cl.run(output, sp.PerfParams.estimate())

    stream.save_mp4('02_masked')
    videos.append('02_masked.mp4')


    # 5. Unbounded State:
    #     This Op will always process all preceding elements of
    #     its input streams before producing a requested output.
    #     This means that sampling operations after this Op
    #     can not change how many inputs it receives. In the next
    #     tutorial, we will show how this can be relaxed for
    #     sub-streams of the input.
    @sp.register_python_op(unbounded_state=True)
    class Example(sp.Kernel):
        def __init__(self, config):
            pass

        def reset(self):
            pass

        def execute(self, frame: sp.FrameType) -> bytes:
            pass

    for stream in streams:
        stream.delete(cl)

    print('Finished! The following videos were written: {:s}'
          .format(', '.join(videos)))


if __name__ == "__main__":
    main()
