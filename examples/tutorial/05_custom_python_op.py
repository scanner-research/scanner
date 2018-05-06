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
@scannerpy.register_python_op(
    inputs=[('frame', ColumnType.Video)],
    outputs=[('resized', ColumnType.Video)])
def resize(columns, config, protobufs):
    # Custom ops take as input three things: the input elements (columns), the kernel configuration (config),
    # and a pointer to the available protobufs (protobufs). Here, we use the width and height from the kernel
    # config to resize our image.
    return [
        cv2.resize(columns[0], (config.args['width'], config.args['height']))
    ]


# If your op has state (e.g. it tracks objects over time) or if it has high start-up costs (e.g. it loads a
# neural network model into memory), then you can also use our class-based interface:


@scannerpy.register_python_op(
    inputs=[('frame', ColumnType.Video)],
    outputs=[('resized', ColumnType.Video)])
class Resize2(Kernel):
    # Init runs once when the class instance is initialized
    def __init__(self, config, protobufs):
        self._width = config.args['width']
        self._height = config.args['height']

    # Execute runs on every element
    def execute(self, columns):
        return [cv2.resize(columns[0], (self._width, self._height))]


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
