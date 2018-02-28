from scannerpy import Database, Job, BulkJob, ColumnType, DeviceType
import os.path

################################################################################
# !! UNDER CONSTRUCTION !! Don't do this tutorial yet                          #
################################################################################

with Database() as db:

    cwd = os.path.dirname(os.path.abspath(__file__))
    db.register_op('MyResize', [('frame', ColumnType.Video)],
                   [('resized', ColumnType.Video)])
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
