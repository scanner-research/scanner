from scannerpy import Database, Job

################################################################################
# This tutorial shows how to select different frames of a video to process.   #
################################################################################

with Database() as db:

    # We can access previously created tables with db.table(name).
    input_table = db.table('example')

    # You can tell Scanner which frames of the video (or which rows of a video
    # table) you want to sample. Here, the "strided" sampling mode will run over
    # every 8th frame, i.e. frames [0, 8, 16, ...]
    frame = input_table.as_op().strided(8)

    # We pass the job to the database same as before, and can process the output
    # same as before.
    histogram = db.ops.Histogram(frame = frame)
    job = Job(columns = [histogram], name = 'example_hist_strided')
    output_table = db.run(job, force=True)

    # Here's some examples of other sampling modes.
    # Range takes a specific subset of a video. Here, it runs over all frames from
    # 0 to 100
    input_table.as_op().range(0, 100)

    # Gather takes an arbitrary list of frames from a video.
    input_table.as_op().gather([10, 17, 32])
