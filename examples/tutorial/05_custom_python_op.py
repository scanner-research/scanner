from scannerpy import Database, Job, ColumnType, DeviceType
import os.path

################################################################################
# This tutorial shows you how to write and use your own Python custom op.      #
################################################################################

db = Database()

# Custom kernels have to be registered with the Scanner runtime, providing their
# name and input/output types as well as op argument paths.
cwd = os.path.dirname(os.path.abspath(__file__))
db.register_op('MyResize', [('frame', ColumnType.Video)],
               [('resized', ColumnType.Video)])

# Custom Python kernels for ops reside in a separate file, here resize_kernel.py.
db.register_python_kernel('MyResize', DeviceType.CPU,
                          cwd + '/resize_kernel.py')

frame = db.sources.FrameColumn()
# Then we use our op just like in the other examples.
resized = db.ops.MyResize(frame=frame, width=200, height=300)
output_op = db.sinks.FrameColumn(columns={'frame': resized})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_resized',
})
db.run(output_op, [job], force=True)
