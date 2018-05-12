from scannerpy.common import *

DEFAULT_GROUP_SIZE = 250

class TaskPartitioner:
    """
    Utility for specifying how to partition the output domain of a job into
    tasks.
    """

    def __init__(self, db):
        self._db = db

    def all(self, group_size=DEFAULT_GROUP_SIZE):
        return self.strided(1, group_size=group_size)

    def strided(self, stride, group_size=DEFAULT_GROUP_SIZE):
        args = self._db.protobufs.StridedPartitionerArgs()
        args.stride = stride
        args.group_size = group_size
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = 'Strided'
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args

    def range(self, start, end):
        return self.ranges([(start, end)])

    def ranges(self, intervals):
        return self.strided_ranges(intervals, 1)

    def gather(self, groups):
        args = self._db.protobufs.GatherSamplerArgs()
        for rows in groups:
            gather_group = args.groups_add()
            gather_group.rows[:] = rows
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = 'Gather'
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args

    def strided_range(self, start, end, stride):
        return self.strided_ranges([(start, end)], stride)

    def strided_ranges(self, intervals, stride):
        args = self._db.protobufs.StridedRangePartitionerArgs()
        args.stride = stride
        for start, end in intervals:
            args.starts.append(start)
            args.ends.append(end)
        sampling_args = self._db.protobufs.SamplingArgs()
        sampling_args.sampling_function = 'StridedRange'
        sampling_args.sampling_args = args.SerializeToString()
        return sampling_args
