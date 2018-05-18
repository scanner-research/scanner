from scannerpy import Database, Job

# The following performs the same computation as main.py, but now on a collection
# of videos instead of just one

db = Database()

# This is the list of videos we are going to process formatted as
# (table_name, video_path). 
videos_to_process = [
    ('sample-clip-1', 'sample-clip-1.mp4'),
    ('sample-clip-2', 'sample-clip-2.mp4'),
    ('sample-clip-3', 'sample-clip-3.mp4')
]

# Ingest the videos into the database
db.ingest_videos(videos_to_process)

# Define the same Computation Graph
frame = db.sources.FrameColumn() # Read from the database
sampled_frame = db.ops.Stride(frame, 3) # Select every third frame
resized = db.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
output_frame = db.sinks.Column(columns={'frame': resized}) # Save resized frame

# Create multiple jobs now, one for each video we want to process
jobs = []
for table_name, _ in videos_to_process:
    job = Job(op_args={
        frame: db.table(table_name).column('frame'), # Column to read input frames from
        output_frame: 'resized-{:s}'.format(table_name) # Name of output table
    })
    jobs.append(job)


# Execute the computation graph and return a handle to the newly produced tables
output_tables = db.run(output=output_frame, jobs=jobs, force=True)

# Save the resized video as an mp4 file
for output_table in output_tables:
    output_table.column('frame').save_mp4(output_table.name())
