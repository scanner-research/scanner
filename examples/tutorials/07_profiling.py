import scannerpy as sp
import scannertools.imgproc

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to look at profiling information for your job.       #
################################################################################

def main():
    cl = sp.Client()

    example_video_path = util.download_video()
    video_stream = sp.NamedVideoStream(cl, 'example', path=example_video_path)
    frames = cl.io.Input([video_stream])

    histograms = cl.ops.Histogram(frame=frames)

    output_stream = sp.NamedVideoStream(cl, 'example_hist_profile')
    output = cl.io.Output(histograms, [output_stream])

    job_id = cl.run(output, sp.PerfParams.estimate())

    # The profiler contains information about how long different parts of your
    # computation take to run. We use Google Chrome's trace format, which you
    # can view by going to chrome://tracing in Chrome and clicking "load" in
    # the top left.
    cl.get_profiler(job_id).write_trace('hist.trace')

    video_stream.delete(cl)
    output_stream.delete(cl)

    # Each row corresponds to a different part of the system, e.g. the thread
    # loading bytes from disk or the thread running your kernels. If you have
    # multiple pipelines or multiple nodes, you will see many of these evaluate
    # threads.

if __name__ == "__main__":
    main()
