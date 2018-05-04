from scannerpy import Database, Job, ColumnType, DeviceType, Kernel
import os
import struct


class MyOpKernel(Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, input_columns):
        return [struct.pack('=q', 9000)]


with Database() as db:
    db.register_op('MyOp', [('frame', ColumnType.Video)], ['test'])
    db.register_python_kernel('MyOp', DeviceType.CPU, MyOpKernel)

    frame = db.sources.FrameColumn()
    test = db.ops.MyOp(frame=frame)
    output = db.sinks.Column(columns={'test': test})

    job = Job(op_args={
        frame: db.table('example').column('frame'),
        output: 'example_py'
    })
    db.run(output=output, jobs=[job])
