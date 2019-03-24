import scannerpy as sp
import scannertools.imgproc

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to select different frames of a video to process.    #
################################################################################

def main():
    cl = sp.Client()

    example_video_path = util.download_video()
    video_stream = sp.NamedVideoStream(cl, 'example', path=example_video_path)

    frames = cl.io.Input([video_stream])

    # You can tell Scanner which frames of the video (or which rows of a video
    # table) you want to sample. Here, we indicate that we want to stride
    # the frame column by 4 (select every 4th frame)
    strided_frames = cl.streams.Stride(frames, [4])

    # We process the sampled frame same as before.
    hists = cl.ops.Histogram(frame=strided_frames)

    hist_stream = sp.NamedVideoStream(cl, 'example_hist_strided')
    output = cl.io.Output(hists, [hist_stream])

    cl.run(output, sp.PerfParams.estimate())

    # Loop over the column's rows. Each row is a tuple of the frame number and
    # value for that row.
    video_hists = hist_stream.load()
    num_rows = 0
    for frame_hists in video_hists:
        assert len(frame_hists) == 3
        assert frame_hists[0].shape[0] == 16
        num_rows += 1
    assert num_rows == round(video_stream.len() / 4)

    video_stream.delete(cl)
    hist_stream.delete(cl)

    # Here's some examples of other sampling modes:

    # Range takes a specific subset of a video. Here, it runs over all frames
    # from 0 to 100
    cl.streams.Range(frames, [(0, 100)])

    # Gather takes an arbitrary list of frames from a video.
    cl.streams.Gather(frames, [[10, 17, 32]])

if __name__ == "__main__":
    main()
