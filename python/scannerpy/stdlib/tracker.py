import cv2
import numpy as np
import scannerpy
import scannerpy.stdlib.bboxes
import scannerpy.stdlib.readers
import scannerpy.stdlib.writers

from scannerpy import Database, Job, DeviceType, FrameType
from scannerpy.stdlib import pipelines
from typing import Sequence

@scannerpy.register_python_op(bounded_state=5)
class TrackObjects(scannerpy.Kernel):
    def __init__(self, config):
        self.config = config

        self.last_merge = []
        self.trackers = []
        self.prev_bboxes = []

    def reset(self):
        self.last_merge = []
        self.trackers = []
        self.prev_bboxes = []

    def execute(self, frame: FrameType, bboxes: bytes) -> bytes:
        # If we have new input boxes, track them
        if bboxes:
            bboxes = scannerpy.stdlib.readers.bboxes(bboxes, self.config.protobufs)
            # Create new trackers for each bbox
            for b in bboxes:
                # If this input box is the same as a tracked box from a previous
                # frame, then we don't need to start a new tracker
                is_same = False
                for i, prev_bbox in enumerate(self.prev_bboxes):
                    if scannerpy.stdlib.bboxes.iou(prev_bbox, b) > 0.25:
                        # We found a match, so ignore this box
                        is_same = True
                        self.last_merge[i] = 0
                        break
                if is_same:
                    continue

                t = cv2.TrackerMIL_create()
                t.init(frame, (b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1))
                self.trackers.append(t)
                self.last_merge.append(0)

        out_bboxes = []
        new_trackers = []
        new_last_merge = []
        for i, t in enumerate(self.trackers):
            self.last_merge[i] += 1
            if self.last_merge[i] > 10:
                continue

            ok, newbox = t.update(frame)
            if ok:
                # Track was successful, so keep the tracker around
                new_trackers.append(t)
                new_last_merge.append(self.last_merge[i])

                # Convert from opencv format to protobuf for serialization
                newbox_proto = self.config.protobufs.BoundingBox()
                newbox_proto.x1 = newbox[0]
                newbox_proto.y1 = newbox[1]
                newbox_proto.x2 = newbox[0] + newbox[2]
                newbox_proto.y2 = newbox[1] + newbox[3]
                out_bboxes.append(newbox_proto)
            else:
                # Tracker failed, so do nothing
                pass

        print('num trackers', len(new_trackers))
        self.trackers = new_trackers
        self.prev_bboxes = out_bboxes
        self.last_merge = new_last_merge

        return scannerpy.stdlib.writers.bboxes(out_bboxes,
                                               self.config.protobufs)
