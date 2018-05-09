from scannerpy import Database, Job
import os.path

################################################################################
# This tutorial shows how to write and use your own C++ custom op.             #
################################################################################

# Look at resize_op/resize_op.cpp to start this tutorial.

db = Database()

cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(cwd, 'resize_op/build/libresize_op.so')):
    print(
        'You need to build the custom op first: \n'
        '$ pushd {}/resize_op; mkdir build && cd build; cmake ..; make; popd'.
        format(cwd))
    exit()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
db.load_op(
    os.path.join(cwd, 'resize_op/build/libresize_op.so'),
    os.path.join(cwd, 'resize_op/build/resize_pb2.py'))

frame = db.sources.FrameColumn()
# Then we use our op just like in the other examples.
resize = db.ops.MyResize(frame=frame, width=200, height=300)
output_op = db.sinks.Column(columns={'resized_frame': resize})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_resized',
})
db.run(output_op, [job], force=True)
