import scannerpy
from scannerpy import Database, Job, DeviceType
import cv2

################################################################################
# This tutorial provides an overview of the more advanced features of Scanner  #
# ops and when you would want to use them.                                     #
################################################################################

db = Database()

# Many ops simply involve applying some processing to their inputs and then
# returning their outputs. But there are also many operations in video
# processing that require the ability to see adjacent frames (such as for
# computing optical flow), need to keep state over time (such as for tracking
# objects), or need to process multiple elements for efficiency reasons (such as
# batching for DNNs).

# Scanner ops therefore have several optional attributes that enable them to
# support these forms of operations:


# 1. Device Type:
#   Ops can specify that they require CPUs or GPUs by declaring their device
#   type. By default, the device_type is DeviceType.CPU.

@scannerpy.register_python_op(device_type=DeviceType.GPU)
def gpu_resize(config, frame: FrameType) -> FrameType:
    return gpu_resize_function(frame)

# 2. Batch:
#   The Op can receive multiple elements at once to enable SIMD or
#   vector-style processing.

@scannerpy.register_python_op(batch=10)
def resize(config, frames: Sequence[FrameType]) -> Sequence[FrameType]:
    output_frames = []
    for frame in frames:
        output_frames.append(cv2.resize(
            frame, (config.args['width'], config.args['height'])))
    return output_frames

frame = db.sources.FrameColumn()
resized_frame = db.ops.resize(frame=frame, batch=10)

# Here we specify that the resize op should receive a batch of 10
# input elements at once. Logically, each element is still processed
# independently but multiple elements are provided for efficient
# batch processing. If there are not enough elements left in a stream,
# the Op may receive less than a batch worth of elements.




# 3. Stencil:
#   The Op requires a window of input elements (for example, the
#   previous and next element) at the same time to produce an
#   output.
@scannerpy.register_python_op(stencil=[0, 1])
def optical_flow(config, frames: Sequence[FrameType]) -> FrameType:
    return cv2.calcOpticalFlowFarneback(frames[0], frames[1])

frame = db.sources.FrameColumn()
flow = db.ops.optical_flow(frame=frame, stencil=[0, 1])



# 4. Bounded State:
#   For each output, the Op requires at least W sequential
#   "warmup" elements before it can produce a valid output.
#   For example, if the output of this Op is sampled
#   sparsely, this guarantees that the Op can "warmup"
#   its state on a stream of W elements before producing the
#   requested output.
@scannerpy.register_python_op(bounded_state=5)
def TrackObjects(scannerpy.Kernel):
    def __init__(self, config):
        self.config = config
        self.trackers = []

    def reset(self):
        self.trackers = []

    def execute(self, frame: FrameType, bbox: bytes) -> bytes:
        # If we have new input boxes, track them
        if bbox:
            bboxes = scannerpy.parsers.bboxes(bbox, self.config.protobufs)
            # Create new trackers for each bbox
            for b in bboxes:
                t = cv2.TrackerMIL_create()
                t.init(frame, (b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1))
                trackers.append(t)

        out_bboxes = []
        new_trackers = []
        for t in trackers:
            ok, newbox = t.update(frame)
            if ok:
                new_trackers.append(t)
                out_bboxes.append(newbox)
            else:
                # Tracker failed, so do nothing
                pass
        self.trackers = new_trackers

        return scannerpy.writers.bboxes(out_bboxes, self.config.protobufs)


# 5. Unbounded State:
#     This Op will always process all preceding elements of
#     its input streams before producing a requested output.
#     This means that sampling operations after this Op
#     can not change how many inputs it receives. In the next
#     tutorial, we will show how this can be relaxed for
#     sub-streams of the input.
@scannerpy.register_python_op(unbounded_state=True)
def Example(scannerpy.Kernel):
    def __init__(self, config):
        pass

    def reset(self):
        pass

    def execute(self, frame: FrameType) -> bytes:
        pass
