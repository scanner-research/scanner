from scannerpy import Database
import numpy as np

# Initialize a connection to the Scanner database. Loads configuration from the
# ~/.scanner.toml configuration file.
db = Database()

# Create an operator to run on our video. This computes a histogram of colors
# for each frame.
hist_op = db.ops.Histogram()

# Create a Scanner table from our video.
input_table = db.ingest_videos([('example', 'example.mp4')])

# Define which frames we're going to run the operator on (all of them, in this
# case). The sampler takes in pairs of (input table name, output table name).
sampler = db.sampler()
tasks = sampler.all([('example', 'example_hist')])

# Run the operator on the input and get an output table.
[output_table] = db.run(tasks, hist_op)

# Load the histograms from a column of the output table. The parse_hist function
# converts the raw bytes output by Scanner into a numpy array for each channel.
def parse_hist(buf):
    return np.split(np.frombuffer(buf, dtype=np.dtype(np.int32)), 3)
video_hists = output_table.columns(0).load(parse_hist)

# Loop over the column's rows. Each row is a tuple of the frame number and
# value for that row.
for (frame, frame_hists) in video_hists:
    assert len(frame_hists) == 3
    print frame, frame_hists[0].shape
