from scannerpy import Database, DeviceType
from scannerpy.stdlib import parsers
import numpy as np
import cv2
import sys
import os.path
sys.path.append(os.path.dirname(__file__) + '/..')
import util

################################################################################
# This file shows a sample end-to-end pipeline that ingests a video into       #
# Scanner, runs a computation, and extracts the results.                       #
################################################################################

# Initialize a connection to the Scanner database. Loads configuration from the
# ~/.scanner.toml configuration file.
db = Database()

# Create an operator to run on our video. This computes a histogram with 16 bins
# for each color channel in a given frame.
hist_op = db.ops.Histogram()

# Create a Scanner table from our video in the format (table name, video path).

# If any videos fail to ingest, they'll show up in the failed list. If force
# is true, it will overwrite existing tables of the same name.
example_video_path = util.download_video()
[input_table], failed = db.ingest_videos([
    ('example', example_video_path),
    ('thisshouldfail', 'thisshouldfail.mp4')], force=True)

print(db.summarize())
print('Failures:', failed)

# Define which frames we're going to run the operator on (all of them, in this
# case). The sampler takes in pairs of (input table name, output table name).
sampler = db.sampler()
tasks = sampler.all([(input_table.name(), 'example_hist')])

# Run the operator on the input and get an output table. The columns of the
# output table are written to disk by the Scanner runtime.
[output_table] = db.run(tasks, hist_op, force=True)

# Load the histograms from a column of the output table. The parsers.histograms
# function  converts the raw bytes output by Scanner into a numpy array for each
# channel.
video_hists = output_table.load(['histogram'], parsers.histograms)

# Loop over the column's rows. Each row is a tuple of the frame number and
# value for that row.
for (frame_index, frame_hists) in video_hists:
    assert len(frame_hists) == 3
    assert frame_hists[0].shape[0] == 16
