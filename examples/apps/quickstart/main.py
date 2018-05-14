from scannerpy import Database, Job

# This is the list of videos we are going to process formatted as
# (table_name, video_path). You can add your own video paths here to process
# them.
videos_to_process = [('example_table', 'sample-clip.mp4')]

# Ingest a video into the database
db = Database()
db.ingest_videos(videos_to_process)

# Define a Computation Graph
frame = db.sources.FrameColumn() # Read from the database
sampled_frame = db.ops.Stride(frame, 3) # Select every third frame
resized = db.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
output_frame = db.sinks.Column(columns={'frame': resized}) # Save resized frame

# If we had multiple input videos, we would create multiple jobs to process in
# parallel
jobs = []
for table_name, _ in videos_to_process:
    job = Job(op_args={
        frame: db.table(table_name).column('frame'), # Column to read input frames from
        output_frame: 'resized_{:s}'.format(table_name) # Name of output table
    })
    jobs.append(job)


# Execute the computation graph and return a handle to the newly produced tables
output_tables = db.run(output=output_frame, jobs=[job], force=True)

# Save the resized video as an mp4 file
output_tables[0].column('frame').save_mp4('resized-video')
