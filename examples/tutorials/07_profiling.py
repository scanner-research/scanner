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

    resized_frames = cl.ops.Resize(frame=frames, width=[640], height=[480])

    output_stream = sp.NamedVideoStream(cl, 'example_profile')
    output = cl.io.Output(resized_frames, [output_stream])

    job_id = cl.run(output, sp.PerfParams.estimate())

    # The profile contains information about how long different parts of your
    # computation take to run. We use Google Chrome's trace format, which you
    # can view by going to chrome://tracing in Chrome and clicking "load" in
    # the top left.
    profile = cl.get_profile(job_id)
    profile.write_trace('resize-graph.trace')
    print('Wrote trace file to "resize-graph.trace".')

    # Each row corresponds to a different part of the system, e.g. the thread
    # loading bytes from disk or the thread running your kernels. If you have
    # multiple pipelines or multiple nodes, you will see many of these evaluate
    # threads.

    video_stream.delete(cl)
    output_stream.delete(cl)


if __name__ == "__main__":
    main()
