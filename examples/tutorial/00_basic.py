from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.stdlib import parsers
import numpy as np
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This file shows a sample end-to-end pipeline that ingests a video into       #
# Scanner, runs a computation, and extracts the results.                       #
################################################################################

# Initialize a connection to the Scanner database. Loads configuration from the
# ~/.scanner.toml configuration file.
with Database() as db:

    # Create a Scanner table from our video in the format (table name,
    # video path). If any videos fail to ingest, they'll show up in the failed
    # list. If force is true, it will overwrite existing tables of the same
    # name.
    example_video_path = util.download_video()
    [input_table], failed = db.ingest_videos([
        ('example', example_video_path),
        ('thisshouldfail', 'thisshouldfail.mp4')], force=True)

    print(db.summarize())
    print('Failures:', failed)

    # Scanner processes videos by forming a graph of operations that operate
    # on input frames from a table and produce outputs to a new table.

    # FrameInput declares that we want to read from a table column that
    # represents a video frame.
    frame = db.ops.FrameInput()

    # These frames are input into a Histogram op that computes a color histogram
    # for each frame.
    hist = db.ops.Histogram(frame=frame)

    # Finally, any columns provided to Output will be saved to the output
    # table at the end of the computation.
    output_op = db.ops.Output(columns=[hist])

    # A job defines a table you want to create. In op_args, we bind the frame
    # input column from above to the table we want to read from and name
    # the output table 'example_hist' by binding a string to output_op.
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            output_op: 'example_hist'
        }
    )
    # Multiple tables can be created using the same execution graph using
    # a bulk job. Here we specify the execution graph (or DAG) by providing
    # the output_op and also specify the jobs we wish to compute.
    bulk_job = BulkJob(output=output_op, jobs=[job])

    # This executes the job and produces the output table. You'll see a progress
    # bar while Scanner is computing the outputs.
    output_tables = db.run(bulk_job, force=True)

    # Load the histograms from a column of the output table. The
    # parsers.histograms  function  converts the raw bytes output by Scanner
    # into a numpy array for each channel.
    video_hists = output_tables[0].load(['histogram'], parsers.histograms)

    # Loop over the column's rows. Each row is a tuple of the frame number and
    # value for that row.
    num_rows = 0
    for (frame_index, frame_hists) in video_hists:
        assert len(frame_hists) == 3
        assert frame_hists[0].shape[0] == 16
        num_rows += 1
    assert num_rows == db.table('example').num_rows()
