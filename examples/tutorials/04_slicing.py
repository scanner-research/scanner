import scannerpy as sp
from typing import Sequence
import cv2
import numpy as np
import subprocess

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to slice streams into sub streams to limit           #
# dependencies between frames.                                                 #
################################################################################

def main():
    sc = sp.Client()

    example_video_path = util.download_video()
    video_stream = sp.NamedVideoStream(sc, 'example', path=example_video_path)

    frame = sc.io.Input([video_stream])

    # When working with bounded or unbounded stateful operations, it is sometimes
    # useful to introduce boundaries between sequences of frames which restrict
    # state being shared between them. For example, if you are tracking objects
    # in a movie, you likely do not want the same trackers when the camera changes
    # scenes since the objects you were tracking are no longer there!

    # Scanner provides support for limiting state propagation across frames through
    # "slicing" operations.
    sliced_frame = sc.streams.Slice(frame, partitions=[sc.partitioner.all(50)])
    # Here, we sliced the input frame stream into chunks of 50 elements. What this
    # means is that any ops which process 'sliced_frame' will *only* be able to
    # maintain state within each chunk of 50 elements.

    # For example, let's say we grab the background subtraction op from the previous
    # tutorial (02_op_attributes) and want to run it on our example video:
    @sp.register_python_op(bounded_state=60)
    class BackgroundSubtraction(sp.Kernel):
        def __init__(self, config, alpha, threshold):
            self.config = config
            self.alpha = alpha
            self.thresh = threshold

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


    frame = sc.io.Input([video_stream])

    # Imagine that there are scene changes at frames 1100, 1200, and 1400, To tell
    # scanner that we do not want background subtraction to cross these boundaries,
    # we can create a 'partitioner' which splits the input.
    scene_partitions = sc.partitioner.ranges([(1100, 1200), (1200, 1400)])

    # Now we slice the input frame sequence into these two partitions using a
    # slice operation
    sliced_frame = sc.streams.Slice(frame, partitions=[scene_partitions])

    # Then we perform background subtraction and indicate we need 60 prior
    # frames to produce correct output
    masked_frame = sc.ops.BackgroundSubtraction(frame=sliced_frame,
                                                alpha=0.02, threshold=0.05,
                                                bounded_state=60)
    # Since the background subtraction operation is done, we can unslice the
    # sequence to join it back into a single contiguous stream. You must unslice
    # sequences before feeding them back into sinks
    unsliced_frame = sc.streams.Unslice(masked_frame)

    stream = sp.NamedVideoStream(sc, '04_masked_video')
    output = sc.io.Output(unsliced_frame, [stream])

    sc.run(output, sp.PerfParams.estimate())

    stream.save_mp4('04_masked')
    stream.delete(sc)

    videos = []
    videos.append('04_masked.mp4')

    print('Finished! The following videos were written: {:s}'
          .format(', '.join(videos)))

if __name__ == "__main__":
    main()
