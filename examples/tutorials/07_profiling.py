from scannerpy import Client, PerfParams
from scannerpy.storage import NamedStream, NamedVideoStream

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to look at profiling information for your job.       #
################################################################################

def main():
    sc = Client()

    example_video_path = util.download_video()
    video_stream = NamedVideoStream(sc, 'example', path=example_video_path)
    frame = sc.io.Input([video_stream])

    histogram = sc.ops.Histogram(frame=frame)

    output_stream = NamedVideoStream(sc, 'example_hist_profile')
    output = sc.io.Output(histogram, [output_stream])

    sc.run(output, PerfParams.estimate())

    # The profiler contains information about how long different parts of your
    # computation take to run. We use Google Chrome's trace format, which you
    # can view by going to chrome://tracing in Chrome and clicking "load" in
    # the top left.
    sc.table('example_hist_profile').profiler().write_trace('hist.trace')

    video_stream.delete(sc)
    output_stream.delete(sc)

    # Each row corresponds to a different part of the system, e.g. the thread
    # loading bytes from disk or the thread running your kernels. If you have
    # multiple pipelines or multiple nodes, you will see many of these evaluate
    # threads.

if __name__ == "__main__":
    main()
