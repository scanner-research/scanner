from scannerpy import Database

################################################################################
# This tutorial shows how to use the Sampler class to select which parts of a  #
# video to process with an op.                                                 #
################################################################################

with Database() as db:
	hist_op = db.ops.Histogram()

	# We can access previously created tables with db.table(name).
	input_table = db.table('example')

	# The sampler lets you run operators over subsets of frames from your videos.
	# Here, the "strided" sampling mode will run over every 8th frame, i.e. frames
	# [0, 8, 16, ...]
	sampler = db.sampler()
	tables = [(input_table.name(), 'example_hist_subsampled')]
	tasks = sampler.strided(tables, 8)

	# We pass the tasks to the database same as before, and can process the output
	# same as before.
	[output_table] = db.run(tasks, hist_op, force=True)

	# Here's some examples of other sampling modes.

	# Range takes a specific subset of a video. Here, it runs over all frames from
	# 0 to 100
	sampler.range(tables, 0, 100)

	# Gather takes an arbitrary list of frames from a video.
	sampler.gather(tables[0], [10, 17, 32])
