import scannerpy as sp
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import util

################################################################################
# This tutorial shows how to write and use new Ops in Python.                  #
################################################################################

# Let's say we want to resize the frames in a video. If Scanner doesn't come
# with a built in "resize" op, we'll have to make our own. There are two ways to
# make a Python op (we also refer to Python ops as "kernels"). The simplest way
# is to write a function that will run independently over each element of the
# input sequence. For example:

# Ops have to be registered with the Scanner runtime, which is done
# here using the decorator @sp.register_python_op()
@sp.register_python_op()
def resize_fn(config, frame: sp.FrameType) -> sp.FrameType:
    # Function ops first input (here, config) is always the kernel config.
    # The kernel config provides metadata about the invocation of the Op,
    # such as:
    # - config.devices: What devices am I allocated to (CPU, GPU, # of devices)
    # - config.args: Arguments provided to the Op when specified in a
    #                computation graph
    # - etc

    # All other inputs must be annotated with one of two types: FrameType for
    # inputs which represent images, or bytes for inputs which represent
    # serialized values. Here we have only one input 'frame' which is of
    # type 'FrameType'.

    # The return type must also be annotated with the same types. For multiple
    # return values, you can specify a tuple of arguments. For example, if
    # we returned both an image and a serialized value, it would be:
    # -> Tuple[FrameType, bytes]

    # Here, we use the width and height from the config args to resize the
    # image.
    return cv2.resize(frame, (config.args['width'], config.args['height']))

# If your op has state (e.g. it tracks objects over time) or if it has high
# start-up costs (e.g. it loads a neural network model into memory), then you
# can also use our class-based interface:
@sp.register_python_op()
class ResizeClass(sp.Kernel):
    # Init runs once when the class instance is initialized
    def __init__(self, config, width, height):
        self._width = width
        self._height = height

    # The execute method serves the same purpose the registered op function
    # above does and has to provide the same type annotations.
    def execute(self, frame: sp.FrameType) -> sp.FrameType:
        return cv2.resize(frame, (self._width, self._height))


def main():
    # Now we can use these new Ops in Scanner:
    sc = sp.Client()

    # Download an example video
    example_video_path = util.download_video()

    # Create a stream and input to read our example video
    video_stream = sp.NamedVideoStream(sc, 'example', path=example_video_path)
    frames = sc.io.Input([video_stream])

    resized_fn_frames = sc.ops.resize_fn(frame=frames, width=640, height=480)

    resized_class_frames = sc.ops.ResizeClass(frame=frames, width=320, height=240)

    fn_stream = sp.NamedVideoStream(sc, 'fn_frames')
    fn_output = sc.io.Output(resized_fn_frames, [fn_stream])

    class_stream = sp.NamedVideoStream(sc, 'class_frames')
    class_output = sc.io.Output(resized_class_frames, [class_stream])

    sc.run([fn_output, class_output], sp.PerfParams.estimate())

    fn_stream.save_mp4('01_resized_fn')
    class_stream.save_mp4('01_resized_class')

    for stream in [fn_stream, class_stream]:
        stream.delete(sc)

    print('Finished! Two videos were saved to the current directory: '
          '01_resized_fn.mp4, 01_resized_class.mp4')

    # If you are trying to integrate with a C++ library or you want a more efficient
    # implementation for your Ops, you can also define Ops in C++. See the
    # 08_defining_cpp_ops.py tutorial.



if __name__ == "__main__":
    main()
