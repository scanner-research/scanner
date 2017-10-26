from scannerpy import Database, Job, DeviceType, BulkJob
import os.path

################################################################################
# This tutorial shows how to write and use your own custom op.                 #
################################################################################

# Look at resize_op/resize_op.cpp to start this tutorial.

with Database() as db:

    if not os.path.isfile('resize_op/build/libresize_op.so'):
        print('You need to build the custom op first: \n'
              '$ cd resize_op; mkdir build && cd build; cmake ..; make')
        exit()

    # To load a custom op into the Scanner runtime, we use db.load_op to open the
    # shared library we compiled. If the op takes arguments, it also optionally
    # takes a path to the generated python file for the arg protobuf.
    db.load_op('resize_op/build/libresize_op.so', 'resize_op/build/resize_pb2.py')

    frame = db.ops.FrameInput()
    # Then we use our op just like in the other examples.
    resize = db.ops.MyResize(
        frame = frame,
        width = 200, height = 300)
    output_op = db.ops.Output(columns=[resize])
    job = Job(
        op_args={
            frame: db.table('example').column('frame'),
            output_op: 'example_resized',
        }
    )
    bulk_job = BulkJob(output=output_op, jobs=[job])
    db.run(bulk_job, force=True)

    print(db.summarize())
