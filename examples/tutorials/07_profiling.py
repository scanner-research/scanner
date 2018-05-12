from scannerpy import Database, Job, DeviceType

################################################################################
# This tutorial shows how to look at profiling information for your job.       #
################################################################################

db = Database()

frame = db.sources.FrameColumn()
histogram = db.ops.Histogram(frame=frame)
output_op = db.sinks.Column(columns={'hist': histogram})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_hist_profile'
})
[output_table] = db.run(output_op, [job], force=True)

# The profiler contains information about how long different parts of your
# computation take to run. We use Google Chrome's trace format, which you
# can view by going to chrome://tracing in Chrome and clicking "load" in
# the top left.
output_table.profiler().write_trace('hist.trace')

# Each row corresponds to a different part of the system, e.g. the thread
# loading bytes from disk or the thread running your kernels. If you have
# multiple pipelines or multiple nodes, you will see many of these evaluate
# threads.
