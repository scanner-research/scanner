from scannerpy import Database, Job, ColumnType, DeviceType
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

with Database() as db:
    db.register_op('MyOp', [('frame', ColumnType.Video)], ['test'])
    kernel_path = script_dir + '/my_kernel.py'
    db.register_python_kernel('MyOp', DeviceType.CPU, kernel_path)

    frame = db.sources.FrameColumn()
    test = db.ops.MyOp(frame = frame)
    output = db.sinks.Column(columns={'test': test})

    job = Job(op_args={
        frame: db.table('example').column('frame'),
        output: 'example_py'
    })
    db.run(output=output, jobs=[job], pipeline_instances_per_node=1)
