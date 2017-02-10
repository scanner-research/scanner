from scannerpy import Database

db = Database()
hist_op = db.ops.Histogram()

# We can access previously created tables with db.table(name).
input_table = db.table('example')

# The sampler lets you run operators over subsets of frames from your videos.
# Here, the "strided" sampling mode will run over every 8th frame.
sampler = db.sampler()
tasks = sampler.strided([('example', 'example_hist_strided')], 8)

# We pass the tasks to the database same as before, and can process the output
# same as before.
[output_table] = db.run(tasks, hist_op)
