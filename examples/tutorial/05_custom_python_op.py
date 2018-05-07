from scannerpy import Database, Job, ColumnType, DeviceType, Kernel
import os.path
import cv2

################################################################################
# This tutorial shows you how to write and use your own Python custom op.      #
################################################################################

# Let's say we want to resize the frames in a video. If Scanner doesn't come with a built in "resize" op,
# we'll have to make our own. You can either do this using C++ (in the next tutorial) for efficiency, or you can
# prototype in Python. There are two ways to make a Python kernel. The simplest way is to write a function that
# will run independently over each element of the input sequence. For example:


# Custom kernels have to be registered with the Scanner runtime, providing their
# name and input/output types as well as op argument paths.
@scannerpy.register_python_op()
def resize(config, frame: FrameType) -> FrameType:
    # Custom ops always take as input the kernel configuration (config). All
    # other inputs must be annotated with one of two types: FrameType for
    # inputs which represent images, or bytes for inputs which represent
    # serialized values. The return type must also be annotated with the same
    # types.

    # Here, we use the width and height from the kernel config to resize our image.
    return cv2.resize(frame, (config.args['width'], config.args['height']))


# If your op has state (e.g. it tracks objects over time) or if it has high start-up costs (e.g. it loads a
# neural network model into memory), then you can also use our class-based interface:
@scannerpy.register_python_op()
class Resize2(Kernel):
    # Init runs once when the class instance is initialized
    def __init__(self, config, protobufs):
        self._width = config.args['width']
        self._height = config.args['height']

    # Execute runs on every element
    def execute(self, frame: FrameType) -> FrameType:
        return cv2.resize(frame, (self._width, self._height))


# Once your ops are ready to go, we boot up the database
db = Database()

# The difference between an op and a kernel is that of interface and implementation. An op represents a kind of
# computation, and a kernel is a concrete implementation of that computation. There can be multiple kernels for
# the same op, e.g. a CPU and GPU implementation.

frame = db.sources.FrameColumn()
# Then we use our op just like in the other examples.
resized = db.ops.MyResize(frame=frame, width=200, height=300)
output_op = db.sinks.FrameColumn(columns={'frame': resized})
job = Job(op_args={
    frame: db.table('example').column('frame'),
    output_op: 'example_resized',
})
db.run(output_op, [job], force=True)
