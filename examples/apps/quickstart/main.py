from scannerpy import Database, Job

db = Database()

# Ingest a video into the database
# The input is formatted as a list of (table_name, video_path). 
db.ingest_videos([('sample-clip', 'sample-clip.mp4')], force=True)

# Define a Computation Graph
frame = db.sources.FrameColumn() # Read from the database
sampled_frame = db.streams.Stride(frame, 3) # Select every third frame
resized = db.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
output_frame = db.sinks.Column(columns={'frame': resized}) # Save resized frame

# Bind arguments to the source and sink nodes
job = Job(op_args={
    frame: db.table(table_name).column('frame'), # Column to read input frames from
    output_frame: 'resized-{:s}'.format(table_name) # Name of output table
})

# Execute the computation graph and return a handle to the newly produced tables
output_tables = db.run(output=output_frame, jobs=[job], force=True)

# Save the resized video as an mp4 file
output_tables[0].column('frame').save_mp4(output_tables[0].name())
