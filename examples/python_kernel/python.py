from scannerpy import Database, Job, BulkJob, ColumnType, DeviceType
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

with Database() as db:
    db.register_op('MyOp', [('frame', ColumnType.Video)], ['test'])
    kernel_path = script_dir + '/my_kernel.py'
    db.register_python_kernel('MyOp', DeviceType.CPU, kernel_path, batch=1)

    frame = db.ops.FrameInput()
    test = db.ops.MyOp(frame = frame, batch = 50)
    output = db.ops.Output(columns=[test])

    job = Job(op_args={
        frame: db.table('example').column('frame'),
        output: 'example_py'
    })
    bulk_job = BulkJob(output=output, jobs=[job])
    db.run(bulk_job, force=True, pipeline_instances_per_node=1)
