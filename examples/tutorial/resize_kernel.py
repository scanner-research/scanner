import scannerpy
import cv2

# A kernel file defines a standalone Python kernel which performs some computation by exporting a
# Kernel class.


class MyResizeKernel(scannerpy.Kernel):
    # __init__ is called once at the creation of the pipeline. Any arguments passed to the kernel
    # are provided through a protobuf object that you manually deserialize. See resize.proto for the
    # protobuf definition.
    def __init__(self, config, protobufs):
        self.args = db.protobufs.MyResizeArgs()
        self.args.ParseFromString(config.args)

    # execute is the core computation routine maps inputs to outputs, e.g. here resizes an input
    # frame to a smaller output frame.
    def execute(self, input_columns):
        return [
            cv2.resize(input_columns[0], (self.args.width, self.args.height))
        ]


KERNEL = MyResizeKernel
