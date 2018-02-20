from scannerpy import Database, Job, BulkJob
from scannerpy.stdlib import parsers

################################################################################
# This tutorial shows how to select different frames of a video to process.   #
################################################################################

with Database() as db:
    frame = db.sources.FrameColumn()

    # You can tell Scanner which frames of the video (or which rows of a video
    # table) you want to sample. Here, we indicate that we want to sample
    # the frame column (we will say how to sample when specifying a job).
    strided_frame = frame.sample()

    # We process the sampled frame same as before.
    hist = db.ops.Histogram(frame=strided_frame)
    output_op = db.ops.Output(columns={'hist': hist})

    # For each job, you can specify how sampling should be performed for
    # a specific column. In the same way we used the op_args argument to bind
    # a table to an input column, we bind a sampling directive to strided_frame.
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            # The "strided" sampling mode will run over # every 8th frame,
            # i.e. frames [0, 8, 16, ...]
            strided_frame: db.sampler.strided(8),
            output_op: 'example_hist_strided'
        }
    )
    bulk_job = BulkJob(output=output_op, jobs=[job])
    output_tables = db.run(bulk_job, force=True, pipeline_instances_per_node=1)

    # Loop over the column's rows. Each row is a tuple of the frame number and
    # value for that row.
    video_hists = output_tables[0].load(['hist'], parsers.histograms)
    num_rows = 0
    for (frame_index, frame_hists) in video_hists:
        assert len(frame_hists) == 3
        assert frame_hists[0].shape[0] == 16
        num_rows += 1
    assert num_rows == db.table('example').num_rows() / 8

    # Here's some examples of other sampling modes.
    # Range takes a specific subset of a video. Here, it runs over all frames
    # from 0 to 100
    db.sampler.range(0, 100)

    # Gather takes an arbitrary list of frames from a video.
    db.sampler.gather([10, 17, 32])
