import scannerpy
import cv2

# A kernel file defines a standalone Python kernel which performs some computation by exporting a
# Kernel class.


class MyResizeKernel(scannerpy.Kernel):
    # __init__ is called once at the creation of the pipeline. Any arguments passed to the kernel
    # are provided through a protobuf object that you manually deserialize. See resize.proto for the
    # protobuf definition.
    def __init__(self, config, protobufs):
        self._width = config.args['width']
        self._height = config.args['height']

    # execute is the core computation routine maps inputs to outputs, e.g. here resizes an input
    # frame to a smaller output frame.
    def execute(self, columns):
        return [cv2.resize(columns[0], (self._width, self._height))]


KERNEL = MyResizeKernel
