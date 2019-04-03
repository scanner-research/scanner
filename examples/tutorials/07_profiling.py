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
    sc = sp.Client()

    example_video_path = util.download_video()
    video_stream = sp.NamedVideoStream(sc, 'example', path=example_video_path)
    frames = sc.io.Input([video_stream])

    resized_frames = sc.ops.Resize(frame=frames, width=[640], height=[480])

    output_stream = sp.NamedVideoStream(sc, 'example_profile')
    output = sc.io.Output(resized_frames, [output_stream])

    job_id = sc.run(output, sp.PerfParams.estimate())

    # The profile contains information about how long different parts of your
    # computation take to run. We use Google Chrome's trace format, which you
    # can view by going to chrome://tracing in Chrome and scicking "load" in
    # the top left.
    profile = sc.get_profile(job_id)
    profile.write_trace('resize-graph.trace')
    print('Wrote trace file to "resize-graph.trace".')

    # Each row corresponds to a different part of the system, e.g. the thread
    # loading bytes from disk or the thread running your kernels. If you have
    # multiple pipelines or multiple nodes, you will see many of these evaluate
    # threads.

    video_stream.delete(sc)
    output_stream.delete(sc)


if __name__ == "__main__":
    main()
