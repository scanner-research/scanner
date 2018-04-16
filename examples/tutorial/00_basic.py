from scannerpy import Database, DeviceType, Job
from scannerpy.stdlib import readers

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
db = Database()

# Create a Scanner table from our video in the format (table name,
# video path). If any videos fail to ingest, they'll show up in the failed
# list. If force is true, it will overwrite existing tables of the same
# name.
example_video_path = util.download_video()
[input_table], failed = db.ingest_videos(
    [('example', example_video_path),
     ('thisshouldfail', 'thisshouldfail.mp4')],
    force=True)

print(db.summarize())
print('Failures:', failed)

# Scanner processes videos by forming a graph of operations that operate
# on input frames from a table and produce outputs to a new table.

# FrameColumn declares that we want to read from a table column that
# represents a video frame.
frame = db.sources.FrameColumn()

# These frames are input into a Histogram op that computes a color histogram
# for each frame.
hist = db.ops.Histogram(frame=frame)

# Finally, any columns provided to Output will be saved to the output
# table at the end of the computation. Here, 'hist' is the name of the
# column for the output table.
output_op = db.sinks.Column(columns={'hist': hist})

# A job defines a table you want to create. In op_args, we bind the
# FrameColumn from above to the table we want to read from and name
# the output table 'example_hist' by binding a string to output_op.
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_hist'
})

# This executes the job and produces the output table. You'll see a progress
# bar while Scanner is computing the outputs.
output_tables = db.run(output=output_op, jobs=[job], force=True)

# Load the histograms from a column of the output table. The
# readers.histograms function converts the raw bytes output by Scanner
# into a numpy array for each channel.
video_hists = output_tables[0].column('hist').load(readers.histograms)

# Loop over the column's values, a set of 3 histograms (for each color channel) per element.
num_rows = 0
for frame_hists in video_hists:
    assert len(frame_hists) == 3
    assert frame_hists[0].shape[0] == 16
    num_rows += 1
assert num_rows == db.table('example').num_rows()
