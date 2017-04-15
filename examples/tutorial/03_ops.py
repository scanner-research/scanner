from scannerpy import Database, Job, DeviceType

################################################################################
# This tutorial shows how to combine multiple operators into a computation     #
# graph and wire inputs/outputs.                                               #
################################################################################

with Database() as db:

    # Scanner can take a directed acyclic graph (DAG) of operators and pass data
    # between them. Each graph has starts with data from an input table.
    frame, frame_info = db.table('example').as_op().all()

    blurred_frame, _ = db.ops.Blur(
        frame = frame,
        frame_info = frame_info,
        kernel_size = 3,
        sigma = 0.5)

    # Multiple operators can be hooked up in a computation by using the outputs
    # of one as the inputs of another.
    histogram = db.ops.Histogram(
        frame = blurred_frame,
        frame_info = frame_info)

    job = Job(
        columns = [histogram],
        name = 'output_table_name')

    db.run(job, force=True)
