from scannerpy import Database, Job, DeviceType

################################################################################
# This tutorial shows how to look at profiling information for your job.       #
################################################################################

with Database() as db:

    frame, frame_info = db.table('example').as_op().all()
    histogram = db.ops.Histogram(frame = frame, frame_info = frame_info)
    job = Job(columns = [histogram], name = 'example_hist_profile')
    output_table = db.run(job, force=True)

    # The profiler contains information about how long different parts of your
    # computation take to run. We use Google Chrome's trace format, which you
    # can view by going to chrome://tracing in Chrome and clicking "load" in
    # the top left.
    output_table.profiler().write_trace('hist.trace')

    # Each row corresponds to a different part of the system, e.g. the thread
    # loading bytes from disk or the thread running your kernels. If you have
    # multiple pipelines or multiple nodes, you will see many of these evaluate
    # threads.
