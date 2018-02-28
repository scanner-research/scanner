from scannerpy import Database, Job, BulkJob, ColumnType, DeviceType
import os.path

################################################################################
# This tutorial shows you how to write and use your own Python custom op.      #
################################################################################

with Database() as db:

    # Custom kernels have to be registered with the Scanner runtime, providing their
    # name and input/output types as well as op argument paths.
    cwd = os.path.dirname(os.path.abspath(__file__))
    db.register_op(
        'MyResize', [('frame', ColumnType.Video)],
        [('resized', ColumnType.Video)],
        proto_path='./resize_pb2.py')

    # Custom Python kernels for ops reside in a separate file, here resize_kernel.py.
    db.register_python_kernel('MyResize', DeviceType.CPU,
                              cwd + '/resize_kernel.py')

    frame = db.ops.FrameInput()
    # Then we use our op just like in the other examples.
    resize = db.ops.MyResize(frame=frame, width=200, height=300)
    output_op = db.ops.Output(columns=[resize])
    job = Job(op_args={
        frame: db.table('example').column('frame'),
        output_op: 'example_resized',
    })
    bulk_job = BulkJob(output=output_op, jobs=[job])
    db.run(bulk_job, force=True)
