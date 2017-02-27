from scannerpy import Database

################################################################################
# This tutorial shows how to combine multiple operators into a computation     #
# graph and wire inputs/outputs.                                               #
################################################################################

db = Database()
sampler = db.sampler()
tasks = sampler.all([('example', 'example_hist_blurred')])

# Scanner can take a directed acyclic graph (DAG) of operators and pass data
# between them. Each graph has an Input node at the beginning that represents
# the data from the input table.
input = db.ops.Input()

# To wire up the graph, you set the inputs of an operator to be the outputs of
# another. Here, the input op outputs two columns, "frame" which is the raw
# bytes of the frame, and "frame_info" which contains information about the
# width/height/etc. of each frame. We feed these two columns into the Blur.
blur = db.ops.Blur(
    inputs=[(input, ["frame", "frame_info"])],
    kernel_size=3,
    sigma=0.5)

# An op can take inputs from multiple other ops, here taking the blurred frame
# from the Blur op and the frame info from the Input op.
hist = db.ops.Histogram(inputs=[(blur, ["frame"]), (input, ["frame_info"])])

# Each op graph must have an Output node at the end that determines which
# columns get saved into the output table.
output = db.ops.Output(inputs=[(hist, ["histogram"])])

# You provide the last op in the graph, here the output op, as the argument to
# db.run.
import timeit
start_time = timeit.default_timer(); db.run(tasks, output, force=True); elapsed = timeit.default_timer() - start_time

# Note: if you don't explicitly include an Input or Output node in your op graph
# they will be automatically added for you. This is how the previous examples
# have worked.
