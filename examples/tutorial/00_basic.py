from scannerpy import Database, DeviceType, Job
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

    # Create a Scanner table from our video in the format (table name, video path).
    # If any videos fail to ingest, they'll show up in the failed list. If force
    # is true, it will overwrite existing tables of the same name.
    example_video_path = util.download_video()
    [input_table], failed = db.ingest_videos([
        ('example', example_video_path),
        ('thisshouldfail', 'thisshouldfail.mp4')], force=True)

    print(db.summarize())
    print('Failures:', failed)

    # To process our video, first we have to define the inputs. The input_table
    # has one column, frame, which we can access via .as_op().
    # The .all() means to include all frames of the video.
    frame = input_table.as_op().all()

    # These frames are input into a Histogram op that computes a color histogram
    # for each frame.
    histogram = db.ops.Histogram(frame = frame)

    # A job defines a table you want to create. Here, we have a single column
    # which is the output of the Histogram op, and we'll name the table
    # 'example_hist'.
    job = Job(columns = [histogram], name = 'example_hist')

    # This executes the job and produces the output table. You'll see a progress
    # bar while Scanner is computing the outputs.
    output_table = db.run(job, force=True)

    # Load the histograms from a column of the output table. The parsers.histograms
    # function  converts the raw bytes output by Scanner into a numpy array for each
    # channel.
    video_hists = output_table.load(['histogram'], parsers.histograms)

    # Loop over the column's rows. Each row is a tuple of the frame number and
    # value for that row.
    for (frame_index, frame_hists) in video_hists:
        assert len(frame_hists) == 3
        assert frame_hists[0].shape[0] == 16
