from scannerpy import Database
from scannerpy.stdlib import parsers
import numpy as np
import cv2

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
([input_table],_) = db.ingest_videos([('example', '/n/scanner/wcrichto.new/videos/movies/private/zootopia_2016.mkv')], force=True)
print(db.summarize())

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
video_hists = output_table.columns(0).load(parsers.histograms)

# Loop over the column's rows. Each row is a tuple of the frame number and
# value for that row.
for (frame_index, frame_hists) in video_hists:
    assert len(frame_hists) == 3
    assert frame_hists[0].shape[0] == 16
