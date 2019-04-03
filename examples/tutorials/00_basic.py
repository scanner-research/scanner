import scannerpy as sp
import scannertools.imgproc

import numpy as np
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This file shows a sample end-to-end pipeline that ingests a video into       #
# Scanner, runs a computation, and extracts the results.                       #
################################################################################

def main():
    # Startup the Scanner runtime and setup a connection to it. Loads configuration from the
    # ~/.scanner.toml configuration file.
    sc = sp.Client()

    example_video_path = util.download_video()

    # Scanner processes videos by forming a graph of operations that operate
    # on input streams and produce output streams. For example, here we can
    # construct a `NamedVideoStream` which reads from an example video:
    video_stream1 = sp.NamedVideoStream(sc, 'example1', path=example_video_path)

    # Now we can start constructing a computation graph. First, we need to declare
    # our input streams that we are going to be reading from. We'll use the 
    # `NamedVideoStream` we just created to build an `Input` operation:
    frames = sc.io.Input([video_stream1])

    # The output of the `Input` op is an edge in the computation graph which represents
    # the sequence of values produced by `Input`, which in this case are frames from
    # the video stream we provided.

    # Now we will process the frames from `Input` using a `Histogram` op that computes
    # a color histogram for each frame.
    hists = sc.ops.Histogram(frame=frames)

    # Finally, we define an output stream to write the computed histograms to.
    # To do this, we will create a `NamedStream` (which is just like a `NamedVideoStream`
    # but for non-video data):
    named_stream1 = sp.NamedStream(sc, 'example1_hist')

    # Then, just like we defined an `Input` op to read the input stream, we'll define
    # an `Output` op to write to the output stream we just defined:
    output_op = sc.io.Output(hists, [named_stream1])

    # Now we can execute this computation graph to produce the output stream.
    # You'll see a progress bar while Scanner is computing the outputs.
    # Note that the .run function also takes as input a PerfParams object which contains some
    # parameters that tune the performance of the job, e.g. how many video frames can fit into memory.
    # By default, you can use PerfParams.estimate() which heuristically guesses an appropriate set of
    # parameters (but is not guaranteed to work!). Later tutorials will address how to tune these params.
    job_id = sc.run(output_op, sp.PerfParams.estimate())

    # Scanner also supports operating over batches of streams to allow for more parallelism.
    # For example, let's define a new graph that operates on two copies of our example video:
    named_stream1.delete(sc)
    video_stream2 = sp.NamedVideoStream(sc, 'example2', path=example_video_path)
    frames = sc.io.Input([video_stream1, video_stream2])
    hists = sc.ops.Histogram(frame=frames)
    named_stream2 = sp.NamedStream(sc, 'example2_hist')
    output_op = sc.io.Output(hists, [named_stream1, named_stream2])

    job_id = sc.run(output_op, sp.PerfParams.estimate())

    # For each of the streams we provided to the one `Input` op in our graph, Scanner will
    # execute the computation graph on the frames from those streams independently. This
    # mechanism allows you to provide Scanner with potentially thousands of videos you
    # would like to process, up front. If Scanner was executing on a cluster of machines,
    # it would be able to parallelize the processing of those videos across the entire cluster.

    # Now that the graph has been processed, we can load the histograms from our computed stream:
    num_rows = 0
    for hist in named_stream1.load():
        assert len(hist) == 3
        assert hist[0].shape[0] == 16
        num_rows += 1
    assert num_rows == video_stream1.len()

    # Just to cleanup, we'll delete the streams we created:
    streams = [video_stream1, video_stream2, named_stream1, named_stream2]
    streams[0].storage().delete(sc, streams)


# It is **CRITICALLY IMPORTANT** that any scripts using Scanner should have their top-level
# (i.e. the file that you call with `python3` from the command line) wrapped in a
# if __name__ == "__main__" block like this. It ensures that Python's multiprocessing library
# works correctly.
if __name__ == "__main__":
    main()
