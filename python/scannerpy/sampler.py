from __future__ import absolute_import, division, print_function, unicode_literals

from scannerpy.common import *

DEFAULT_TASK_SIZE = 250

class Sampler:
    """
    Utility for specifying which frames of a video (or which rows of a table)
    to run a computation over.
    """

    def __init__(self, db):
        self._db = db

    def all(self):
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = "All"
        return sampling_args

    def strided(self, stride):
        args = self._db.protobufs.StridedSamplerArgs()
        args.stride = stride
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = "Strided"
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args

    def range(self, start, end):
        return self.ranges([(start, end)])

    def ranges(self, intervals):
        return self.strided_ranges(intervals, 1)

    def gather(self, rows):
        args = self._db.protobufs.GatherSamplerArgs()
        args.rows[:] = rows
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = 'Gather'
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args

    def strided_range(self, start, end, stride):
        return self.strided_ranges([(start, end)], stride)

    def strided_ranges(self, intervals, stride):
        args = self._db.protobufs.StridedRangeSamplerArgs()
        args.stride = stride
        for start, end in intervals:
            args.starts.append(start)
            args.ends.append(end)
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = "StridedRanges"
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args

    def space_null(self, spacing):
        args = self._db.protobufs.SpaceNullSamplerArgs()
        args.spacing = spacing
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = "SpaceNull"
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args

    def space_repeat(self, spacing):
        args = self._db.protobufs.SpaceRepeatSamplerArgs()
        args.spacing = spacing
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = "SpaceRepeat"
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args
