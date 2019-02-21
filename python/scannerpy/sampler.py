from scannerpy.common import *
from scannerpy.protobufs import protobufs

DEFAULT_TASK_SIZE = 125

class Sampler:
    """
    Utility for specifying which frames of a video (or which rows of a table)
    to run a computation over.
    """

    def __init__(self, sc):
        self._sc = sc

    def All(self, input):
        def arg_builder():
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "All"
            return sampling_args
        return self._sc.ops.Sample(
            col=self,
            args={'arg_builder': arg_builder,
                  'default': arg_builder()})

    def Stride(self, input, stride=None):
        def arg_builder(stride=stride):
            args = protobufs.StridedSamplerArgs()
            args.stride = stride
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "Strided"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._sc.ops.Sample(
            col=self,
            args={'arg_builder': arg_builder,
                  'default': arg_builder() if stride else None})

    def Range(self, input, start=None, end=None):
        return self.ranges([(start, end)])

    def Ranges(self, input, intervals=None):
        return self.strided_ranges(intervals, 1)

    def Gather(self, input, rows=None):
        def arg_builder(rows=rows):
            args = protobufs.GatherSamplerArgs()
            args.rows[:] = rows
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = 'Gather'
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._sc.ops.Sample(
            col=self,
            args={'arg_builder': arg_builder,
                  'default': arg_builder() if rows else None})


    def StridedRange(self, input, start=None, end=None, stride=None):
        return self.strided_ranges([(start, end)], stride)

    def StridedRanges(self, input, intervals=None, stride=None):
        def arg_builder(intervals=intervals, stride=stride):
            args = protobufs.StridedRangeSamplerArgs()
            args.stride = stride
            for start, end in intervals:
                args.starts.append(start)
                args.ends.append(end)
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._sc.ops.Sample(
            col=self,
            args={'arg_builder': arg_builder,
                  'default': arg_builder() if intervals and stride else None})

    def RepeatNull(self, input, spacing=None):
        def arg_builder(spacing=spacing):
            args = protobufs.SpaceNullSamplerArgs()
            args.spacing = spacing
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "SpaceNull"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return sampling_args
        return self._sc.ops.Space(
            col=self,
            args={'arg_builder': arg_builder,
                  'default': arg_builder() if spacing else None})

    def Repeat(self, input, spacing=None):
        def arg_builder(spacing=spacing):
            args = protobufs.SpaceRepeatSamplerArgs()
            args.spacing = spacing
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "SpaceRepeat"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._sc.ops.Space(
            col=self,
            args={'arg_builder': arg_builder,
                  'default': arg_builder() if spacing else None})
