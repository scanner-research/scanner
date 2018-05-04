import scannerpy
import scannerpy.stdlib.readers as readers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.bboxes as bboxes


class BBoxNMSKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        self.scale = config.args['scale']

    def close(self):
        pass

    def execute(self, input_columns):
        bboxes_list = []
        for c in input_columns:
            bboxes_list += readers.bboxes(c, self.protobufs)
        nmsed_bboxes = bboxes.nms(bboxes_list, 0.1)
        return [writers.bboxes(nmsed_bboxes, self.protobufs)]


KERNEL = BBoxNMSKernel
