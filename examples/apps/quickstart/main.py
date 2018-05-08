from scannerpy import Database, Job

# Ingest a video into the database
db = Database()
db.ingest_videos([('example_table', 'sample-clip.mp4')])

# Define a Computation Graph
frame = db.sources.FrameColumn() # Read from the database
sampled_frame = frame.sample() # Select only some of the frames
resized = db.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
output_frame = db.sinks.Column(columns={'frame': resized}) # Save resized frame

job = Job(op_args={
    frame: db.table('example_table').column('frame'), # Column to read input frames from
    sampled_frame: db.sampler.stride(3), # Sample every 3rd frame
    output_frame: 'resized_example' # Name of output table
})

# Execute the computation graph and return a handle to the newly produced tables
output_tables = db.run(output=output_frame, jobs=[job], force=True)

# Save the resized video as an mp4 file
output_tables[0].column('frame').save_mp4('resized-video')
