from scannerpy import Database, TableJob

################################################################################
# This tutorial shows how to use the Sampler class to select which parts of a  #
# video to process with an op.                                                 #
################################################################################

with Database() as db:

    # We can access previously created tables with db.table(name).
    input_table = db.table('example')

    frame, frame_info = input_table.as_op().strided(8)
    histogram = db.ops.Histogram(frame = frame, frame_info = frame_info)
    job = TableJob(columns=[histogram], name='example_hist_strided')

    # The sampler lets you run operators over subsets of frames from your videos.
    # Here, the "strided" sampling mode will run over every 8th frame, i.e. frames
    # [0, 8, 16, ...]

    # We pass the tasks to the database same as before, and can process the output
    # same as before.
    [output_table] = db.run([job], force=True)

    # Here's some examples of other sampling modes.

    # Range takes a specific subset of a video. Here, it runs over all frames from
    # 0 to 100
    input_table.as_op().range(0, 100)

    # Gather takes an arbitrary list of frames from a video.
    input_table.as_op().gather([10, 17, 32])
