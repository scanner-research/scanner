import scannerpy
import scannerpy.stdlib.parsers as parsers
import scannerpy.stdlib.writers as writers
import scannerpy.stdlib.bboxes as bboxes

class BBoxNMSKernel(scannerpy.Kernel):
    def __init__(self, config, protobufs):
        self.protobufs = protobufs
        args = protobufs.BBoxNMSArgs()
        args.ParseFromString(config)
        self.scale = args.scale

    def close(self):
        pass

    def execute(self, input_columns):
        bboxes_list = []
        for c in input_columns:
            bboxes_list += parsers.bboxes(c, self.protobufs)
        nmsed_bboxes = bboxes.nms(bboxes_list, 0.1)
        return writers.bboxes([nmsed_bboxes], self.protobufs)

KERNEL = BBoxNMSKernel
