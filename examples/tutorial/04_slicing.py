from scannerpy import Database, Job, BulkJob, DeviceType
from scannerpy.stdlib import parsers
import math

################################################################################
# This tutorial shows how to use column slicing to limit Op dependencies       #
# within subsequences of the input.                                            #
################################################################################

with Database(debug=True) as db:
    frame = db.ops.FrameInput()

    # You can tell Scanner which frames of the video (or which rows of a video
    # table) you want to sample. Here, we indicate that we want to sample
    # the frame column (we will say how to sample when specifying a job).
    sliced_frame = frame.slice()

    # We process the sampled frame same as before.
    hist = db.ops.Histogram(frame=sliced_frame)
    unsliced_hist = hist.unslice()

    output_op = db.ops.Output(columns=[unsliced_hist])

    # For each job, you can specify how sampling should be performed for
    # a specific column. In the same way we used the op_args argument to bind
    # a table to an input column, we bind a sampling directive to strided_frame.
    job = Job(
        output_table_name='example_hist_sliced',
        op_args={
            frame: db.table('example').column('frame'),
            # The "strided" sampling mode will run over # every 8th frame,
            # i.e. frames [0, 8, 16, ...]
            sliced_frame: db.partitioner.all(500),
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    output_tables = db.run(bulk_job, force=True, pipeline_instances_per_node=2)

    # Loop over the column's rows. Each row is a tuple of the frame number and
    # value for that row.
    video_hists = output_tables[0].load(['histogram'], parsers.histograms)
    num_rows = 0
    for (frame_index, frame_hists) in video_hists:
        assert len(frame_hists) == 3
        assert frame_hists[0].shape[0] == 16
        num_rows += 1
    print(num_rows)
    assert num_rows == db.table('example').num_rows()

    #
    frame = db.ops.FrameInput()
    sliced_frame = frame.slice()
    hist = db.ops.Histogram(frame=sliced_frame)

    gath_hist = hist.sample()

    unsliced_hist = gath_hist.unslice()
    output_op = db.ops.Output(columns=[unsliced_hist])

    # For each job, you can specify how sampling should be performed for
    # a specific column. In the same way we used the op_args argument to bind
    # a table to an input column, we bind a sampling directive to strided_frame.
    num_slice_groups = int(math.ceil(db.table('example').num_rows() / 500.0))
    job = Job(
        output_table_name='example_hist_sliced_gath',
        op_args={
            frame: db.table('example').column('frame'),
            sliced_frame: db.partitioner.all(500),
            gath_hist: [db.sampler.gather([0, 15])
                        for _ in range(num_slice_groups)]
        }
    )
    bulk_job = BulkJob(dag=output_op, jobs=[job])
    output_tables = db.run(bulk_job, force=True, pipeline_instances_per_node=2)

    # Loop over the column's rows. Each row is a tuple of the frame number and
    # value for that row.
    video_hists = output_tables[0].load(['histogram'], parsers.histograms)
    num_rows = 0
    for (frame_index, frame_hists) in video_hists:
        assert len(frame_hists) == 3
        assert frame_hists[0].shape[0] == 16
        num_rows += 1
    assert num_rows == num_slice_groups * 2
