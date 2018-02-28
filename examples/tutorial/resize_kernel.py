import scannerpy
import cv2


class MyResizeKernel(scannerpy.Kernel):
    def __init__(self, config):
        print config

    def execute(self, input_columns):
        return cv2.resize(input_columns[0], (100, 100))


KERNEL = MyResizeKernel
