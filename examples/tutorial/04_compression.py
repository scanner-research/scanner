from scannerpy import Database, Job, DeviceType

################################################################################
# This tutorial discusses how Scanner compresses output columns, how to        #
# control how and when this compression happens, and how to export compressed  #
# video files.
################################################################################

db = Database()


# Frames on disk can either be stored uncompressed (raw bits) or compressed
# (encoded using some form of image or video compression). When Scanner
# reads frames from a table, it automatically decodes the data if necessary.
# The Op DAG only sees the raw frames. For example, this table is stored
# as compressed video.
def make_blurred_frame():
    frame = db.sources.FrameColumn()

    blurred_frame = db.ops.Blur(frame=frame, kernel_size=3, sigma=0.5)
    return frame, blurred_frame


# By default, if an Op outputs a frame with 3 channels with type uint8,
# those frames will be compressed using video encoding. No other frame
# type is currently compressed.
frame, blurred_frame = make_blurred_frame()
output_op = db.sinks.Column(columns={'frame': blurred_frame})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'output_table_name',
})
db.run(output_op, [job], force=True)

frame, blurred_frame = make_blurred_frame()
# The compression parameters can be controlled by annotating the column
low_quality_frame = blurred_frame.compress_video(quality=35)
output_op = db.sinks.Column(columns={'frame': low_quality_frame})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'low_quality_table',
})
db.run(output_op, [job], force=True)

# If no compression is desired, this can be specified by indicating that
# the column should be lossless.
frame, blurred_frame = make_blurred_frame()
# The compression parameters can be controlled by annotating the column
lossless_frame = blurred_frame.lossless()
output_op = db.sinks.Column(columns={'frame': lossless_frame})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'pristine_frame',
})
db.run(output_op, [job], force=True)

# Any column which is saved as compressed video can be exported as an mp4
# file by calling save_mp4 on the column. This will output a file called
# 'low_quality_video.mp4' in the current directory.
db.table('low_quality_table').column('frame').save_mp4('low_quality_video')
