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
    cl = sp.Client()

    # Create a Scanner stream from our video in the format (table name,
    # video path). If any videos fail to ingest, they'll show up in the failed
    # list. If force is true, it will overwrite existing tables of the same
    # name.
    example_video_path = util.download_video()
    _, failed = cl.ingest_videos(
        [('example1', example_video_path),
         ('thisshouldfail', 'thisshouldfail.mp4')],
        force=True)

    print(cl.summarize())
    print('Failures:', failed)

    # Scanner processes videos by forming a graph of operations that operate
    # on input streams and produce output streams. For example, here we can
    # construct a `NamedVideoStream` which reads from the video we just
    # ingested into Scanner:
    video_stream1 = sp.NamedVideoStream(cl, 'example1')

    # `NamedVideoStream`s also support directly reading video files (the ingest process
    # will occur automatically).
    video_stream2 = sp.NamedVideoStream(cl, 'example2', path=example_video_path)

    # Now we can start constructing a computation graph. First, we need to declare
    # our input streams that we are going to be reading from. We'll use the two
    # `NamedVideoStream`s we just created to build an `Input` operation:
    frame = cl.io.Input([video_stream1, video_stream2])
    # The output of the `Input` op is an edge in the computation graph which represents
    # the sequence of values produced by `Input`, which in this case are frames from
    # the two videos we provided.

    # Now we will process the frames from `Input` using a `Histogram` op that computes
    # a color histogram for each frame.
    hist = cl.ops.Histogram(frame=frame)

    # Finally, we want define an output stream to write the computed histograms to.
    # To do this, we # will create two `NamedStream`s (which are just like a
    # `NamedVideoStream` but for non-video data), one for each input stream:
    named_stream1 = sp.NamedStream(cl, 'example1_hist')
    named_stream2 = sp.NamedStream(cl, 'example2_hist')
    # Then, just like we defined an `Input` op to read the input streams,  we'll define
    # an `Output` op to write to the output streams we just defined:
    output_op = cl.io.Output(hist, [named_stream1, named_stream2])

    # Now we can execute this computation graph to compute the output streams.
    # You'll see a progress bar while Scanner is computing the outputs.
    # Note that the .run function also takes as input a PerfParams object which contains some
    # parameters that tune the performance of the job, e.g. how many video frames can fit into memory.
    # By default, you can use PerfParams.estimate() which heuristically guesses an appropriate set of
    # parameters (but is not guaranteed to work!). Later tutorials will address how to tune these params.
    cl.run(output_op, sp.PerfParams.estimate())
    # For each of the streams we provided to the one `Input` op in our graph, Scanner will
    # execute the computation graph on the frames from those streams independently. This
    # mechanism allows you to provide Scanner with potentially thousands of videos you
    # would like to process, up front. If Scanner was executing on a cluster of machines,
    # it would be able to parallelize the processing of those videos across the entire cluster.

    print(cl.summarize())

    # Now that the graph has been processed, we can load the histograms from our computed stream:
    num_rows = 0
    for hist in named_stream1.load():
        assert len(hist) == 3
        assert hist[0].shape[0] == 16
        num_rows += 1
    assert num_rows == video_stream1.len()

    # Just to cleanup, we'll delete the streams we created:
    streams = [video_stream1, video_stream2, named_stream1, named_stream2]
    streams[0].storage().delete(cl, streams)


# It is **CRITICALLY IMPORTANT** that any scripts using Scanner should have their top-level
# (i.e. the file that you call with `python3` from the command line) wrapped in a
# if __name__ == "__main__" block like this. It ensures that Python's multiprocessing library
# works correctly.
if __name__ == "__main__":
    main()
