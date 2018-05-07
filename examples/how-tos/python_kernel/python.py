import scannerpy
import os
import struct

from scannerpy import Database, Job, FrameType, DeviceType, Kernel
from typing import Tuple

@scannerpy.register_python_op()
class MyOpKernel(Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs

    def close(self):
        pass

    def execute(self, frame: FrameType) -> bytes:
        return struct.pack('=q', 9000)


with Database() as db:
    frame = db.sources.FrameColumn()
    test = db.ops.MyOp(frame=frame)
    output = db.sinks.Column(columns={'test': test})

    job = Job(op_args={
        frame: db.table('example').column('frame'),
        output: 'example_py'
    })
    db.run(output=output, jobs=[job])
