from scannerpy import Database, Job, DeviceType

################################################################################
# This tutorial discusses how Scanner compresses output columns, how to        #
# control how and when this compression happens, and how to export compressed  #
# video files.
################################################################################

with Database() as db:

    # Frames on disk can either be stored uncompressed (raw bits) or compressed
    # (encoded using some form of image or video compression). When Scanner
    # reads frames from a table, it automatically decodes the data if necessary.
    # The Op DAG only sees the raw frames. For example, this table is stored
    # as compressed video.
    def make_blurred_frame():
        frame = db.table('example').as_op().all()

        blurred_frame = db.ops.Blur(
            frame = frame,
            kernel_size = 3,
            sigma = 0.5)
        return blurred_frame

    blurred_frame = make_blurred_frame()

    # By default, if an Op outputs a frame with 3 channels with type uint8,
    # those frames will be compressed using video encoding. No other frame
    # type is currently compressed.
    job = Job(
        columns = [blurred_frame],
        name = 'output_table_name')
    db.run(job, force=True)

    # The compression parameters can be controlled by annotating the column
    blurred_frame = make_blurred_frame()
    low_quality_frame = blurred_frame.compress_video(quality = 35)

    job = Job(
        columns = [low_quality_frame],
        name = 'low_quality_table')
    db.run(job, force=True)

    # If no compression is desired, this can be specified by indicating that
    # the column should be lossless.
    blurred_frame = make_blurred_frame()
    lossless_frame = blurred_frame.lossless()

    job = Job(
        columns = [lossless_frame],
        name = 'pristine_frame')
    db.run(job, force=True)

    # Any column which is saved as compressed video can be exported as an mp4
    # file by calling save_mp4 on the column. This will output a file called
    # 'low_quality_video.mp4' in the current directory.
    db.table('low_quality_table').column('frame').save_mp4('low_quality_video')
