from scannerpy.common import *

DEFAULT_TASK_SIZE = 125

class StreamsGenerator:
    """
    Utility for specifying which frames of a video (or which rows of a table)
    to run a computation over.
    """

    def __init__(self, db):
        self._db = db

    def Slice(self, input, partitioner=None):
        def arg_builder(partitioner=partitioner):
            return partitioner
        return self._db.ops.Slice(
            col=input,
            extra={'type': 'Slice',
                   'arg_builder': arg_builder,
                   'default': partitioner})

    def Unslice(self, input):
        return self._db.ops.Unslice(col=input)

    def All(self, input):
        def arg_builder():
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "All"
            return sampling_args
        return self._db.ops.Sample(
            col=input,
            extra={'type': 'All',
                   'arg_builder': arg_builder,
                   'default': {}})

    def Stride(self, input, stride=None):
        def arg_builder(stride=stride):
            args = self._db.protobufs.StridedSamplerArgs()
            args.stride = stride
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "Strided"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Stride',
                   'arg_builder': arg_builder,
                   'default': stride})

    def Range(self, input, start=None, end=None):
        def arg_builder(start=start, end=end):
            args = self._db.protobufs.StridedRangeSamplerArgs()
            args.stride = 1
            args.starts.append(start)
            args.ends.append(end)
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Range',
                   'arg_builder': arg_builder,
                   'default': (start, end) if (start is not None and
                                               end is not None) else None})


    def Ranges(self, input, intervals=None):
        def arg_builder(intervals=intervals):
            args = self._db.protobufs.StridedRangeSamplerArgs()
            args.stride = 1
            for start, end in intervals:
                args.starts.append(start)
                args.ends.append(end)
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Ranges',
                   'arg_builder': arg_builder,
                   'default': intervals if intervals else None})

    def StridedRange(self, input, start=None, end=None, stride=None):
        def arg_builder(start=start, end=end, stride=stride):
            args = self._db.protobufs.StridedRangeSamplerArgs()
            args.stride = stride
            args.starts.append(start)
            args.ends.append(end)
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'StridedRange',
                   'arg_builder': arg_builder,
                   'default': (start, end, stride) if (start is not None and
                                                       end is not None and
                                                       stride is not None) else None})


    def StridedRanges(self, input, intervals=None, stride=None):
        def arg_builder(intervals=intervals, stride=stride):
            args = self._db.protobufs.StridedRangeSamplerArgs()
            args.stride = stride
            for start, end in intervals:
                args.starts.append(start)
                args.ends.append(end)
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'StridedRanges',
                   'arg_builder': arg_builder,
                   'default': (intervals, stride) if (intervals is not None and
                                                      stride is not None) else None})

    def Gather(self, input, rows=None):
        def arg_builder(rows=rows):
            args = self._db.protobufs.GatherSamplerArgs()
            args.rows[:] = rows
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = 'Gather'
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Gather',
                   'arg_builder': arg_builder,
                   'default': {'rows': rows}})

    def RepeatNull(self, input, spacing=None):
        def arg_builder(spacing=spacing):
            args = self._db.protobufs.SpaceNullSamplerArgs()
            args.spacing = spacing
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "SpaceNull"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Space(
            col=input,
            extra={'type': 'RepeatNull',
                   'arg_builder': arg_builder,
                   'default': spacing})

    def Repeat(self, input, spacing=None):
        def arg_builder(spacing=spacing):
            args = self._db.protobufs.SpaceRepeatSamplerArgs()
            args.spacing = spacing
            sampling_args = self._db.protobufs.SamplingArgs()
            sampling_args.sampling_function = "SpaceRepeat"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Space(
            col=input,
            extra={'type': 'Repeat',
                   'arg_builder': arg_builder,
                   'default': spacing})
