from scannerpy import Client, PerfParams
from scannerpy.storage import NamedStream, NamedVideoStream
import scannertools

import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial discusses how Scanner compresses output columns, how to        #
# control how and when this compression happens, and how to export compressed  #
# video files.
################################################################################

def main():
    sc = Client()

    # Frames on disk can either be stored uncompressed (raw bits) or compressed
    # (encoded using some form of image or video compression). When Scanner
    # reads frames from a table, it automatically decodes the data if necessary.
    # The Op DAG only sees the raw frames. For example, this table is stored
    # as compressed video.
    def make_blurred_frame(streams):
        frame = sc.io.Input(streams)
        blurred_frame = sc.ops.Blur(frame=frame, kernel_size=3, sigma=0.5)
        sampled_frame = sc.streams.Range(blurred_frame, [(0, 30)])
        return frame, sampled_frame

    example_video_path = util.download_video()
    video_stream = NamedVideoStream(sc, 'example', path=example_video_path)

    # By default, if an Op outputs a frame with 3 channels with type uint8,
    # those frames will be compressed using video encoding. No other frame
    # type is currently compressed.
    frame, blurred_frame = make_blurred_frame([video_stream])

    stream = NamedVideoStream(sc, 'output_table_name')
    output = sc.io.Output(blurred_frame, [stream])
    sc.run(output)

    stream.delete(sc)

    frame, blurred_frame = make_blurred_frame([video_stream])
    # The compression parameters can be controlled by annotating the output
    # of an Op that produces frames
    low_quality_frame = blurred_frame.compress_video(quality=35)

    low_quality_stream = NamedVideoStream(sc, 'low_quality_video')
    output = sc.io.Output(low_quality_frame, [low_quality_stream])
    sc.run(output, PerfParams.estimate())

    frame, blurred_frame = make_blurred_frame([video_stream])
    # If no compression is desired, this can be specified by indicating that
    # the Op output should be lossless.
    lossless_frame = blurred_frame.lossless()

    lossless_stream = NamedVideoStream(sc, 'lossless_video')
    output = sc.io.Output(lossless_frame, [lossless_stream])

    sc.run(output)

    # Any sequence of frames which are saved as a compressed `NamedVideoStream` can
    # be exported as an mp4 file by calling save_mp4 on the stream. This will output
    # a file called 'low_quality_video.mp4' in the current directory.
    low_quality_stream.save_mp4('low_quality_video')

    low_quality_stream.delete(sc)
    lossless_stream.delete(sc)

if __name__ == "__main__":
    main()
