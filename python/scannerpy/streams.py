import scannerpy.op

from scannerpy.common import *
from scannerpy.protobufs import protobufs
from scannerpy.op import SliceList
from typing import Sequence, Union, Tuple, Optional, List

class StreamsGenerator:
    r"""Provides Ops for sampling elements from streams.

    The methods of this class construct Scanner Ops that enable selecting
    subsets of the elements in a stream to produce new streams.

    This class should not be constructed directly, but accessed via a Database
    object like:

      db.streams.Range(input)
    """

    def __init__(self, db):
        self._db = db

    def Slice(self,
              input: scannerpy.op.OpColumn,
              partitions) -> scannerpy.op.OpColumn:
        r"""Partitions a stream into independent substreams.

        Parameters
        ----------
        input
          The stream to partition.

        partitioner
          The partitioner that should be used to split the stream into
          substreams.

        Returns
        -------
        scannerpy.op.OpColumn
          A new stream which represents multiple substreams.
        """
        def arg_builder(partitioner):
            return partitioner
        return self._db.ops.Slice(
            col=input,
            extra={'type': 'Slice',
                   'arg_builder': arg_builder,
                   'job_args': partitions})

    def Unslice(self, input: scannerpy.op.OpColumn) -> scannerpy.op.OpColumn:
        r"""Joins substreams back together.

        Parameters
        ----------
        input
          The stream which contains substreams to join back together.

        Returns
        -------
        scannerpy.op.OpColumn
          A new stream which is the concatentation of the input substreams.
        """
        return self._db.ops.Unslice(col=input)

    def All(self, input: scannerpy.op.OpColumn) -> scannerpy.op.OpColumn:
        r"""Samples all elements from the stream.

        Serves as an identity sampling function.

        Parameters
        ----------
        input
          The stream to sample.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder():
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "All"
            return sampling_args
        return self._db.ops.Sample(
            col=input,
            extra={'type': 'All',
                   'arg_builder': arg_builder,
                   'default': {}})

    def Stride(self,
               input: scannerpy.op.OpColumn,
               strides) -> scannerpy.op.OpColumn:
        r"""Samples every n'th element from the stream, where n is the stride.

        Parameters
        ----------
        input
          The stream to sample.

        stride
          The default value to stride by for all jobs.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder(stride):
            args = protobufs.StridedSamplerArgs()
            args.stride = stride
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "Strided"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Stride',
                   'arg_builder': arg_builder,
                   'job_args': strides})

    def Range(self,
              input: scannerpy.op.OpColumn,
              ranges) -> scannerpy.op.OpColumn:
        r"""Samples a range of elements from the input stream.

        Parameters
        ----------
        input
          The stream to sample.

        start
          The default index to start sampling from.

        end
          The default index to end sampling at.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder(start, end):
            args = protobufs.StridedRangeSamplerArgs()
            args.stride = 1
            args.starts.append(start)
            args.ends.append(end)
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Range',
                   'arg_builder': arg_builder,
                   'job_args': ranges})

    def Ranges(self,
               input: scannerpy.op.OpColumn,
               intervals: List[Sequence[Tuple[int, int]]]) -> scannerpy.op.OpColumn:
        r"""Samples multiple ranges of elements from the input stream.

        Parameters
        ----------
        input
          The stream to sample.

        intervals
          The default intervals to sample from. This should be a list
          of tuples representing start and end ranges.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.

        Examples
        --------
        For example, to select frames 0-10 and 100-200, you would write:

        db.streams.Ranges(input=input, intervals=[(0, 11), (100, 201)])
        """
        def arg_builder(intervals):
            args = protobufs.StridedRangeSamplerArgs()
            args.stride = 1
            for start, end in intervals:
                args.starts.append(start)
                args.ends.append(end)
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Ranges',
                   'arg_builder': arg_builder,
                   'job_args': intervals})

    def StridedRange(self,
                     input: scannerpy.op.OpColumn,
                     ranges) -> scannerpy.op.OpColumn:
        r"""Samples a strided range of elements from the input stream.

        Parameters
        ----------
        input
          The stream to sample.

        start
          The default index to start sampling from.

        end
          The default index to end sampling at.

        stride
          The default value to stride by.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder(start, end, stride):
            args = protobufs.StridedRangeSamplerArgs()
            args.stride = stride
            args.starts.append(start)
            args.ends.append(end)
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "StridedRanges"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'StridedRange',
                   'arg_builder': arg_builder,
                   'job_args': ranges})


    def StridedRanges(self,
                      input: scannerpy.op.OpColumn,
                      intervals: Sequence[Tuple[int, int]] = None,
                      stride: int = None) -> scannerpy.op.OpColumn:
        r"""Samples strided ranges of elements from the input stream.

        Parameters
        ----------
        input
          The stream to sample.

        intervals
          The default intervals to sample from. This should be a list
          of tuples representing start and end ranges.

        stride
          The default value to stride by.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
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

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'StridedRanges',
                   'arg_builder': arg_builder,
                   'default': (intervals, stride) if (intervals is not None and
                                                      stride is not None) else None})

    def Gather(self,
               input: scannerpy.op.OpColumn,
               indices: Sequence[Sequence[int]]) -> scannerpy.op.OpColumn:
        r"""Samples a list of elements from the input stream.

        Parameters
        ----------
        input
          The stream to sample.

        rows
          A list of the indices to sample.

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder(rows):
            args = protobufs.GatherSamplerArgs()
            args.rows[:] = rows
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = 'Gather'
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Sample(
            col=input,
            extra={'type': 'Gather',
                   'arg_builder': arg_builder,
                   'job_args': indices})


    def RepeatNull(self,
                   input: scannerpy.op.OpColumn,
                   spacings) -> scannerpy.op.OpColumn:
        r"""Expands a sequence by inserting nulls.

        Parameters
        ----------
        input
          The stream to expand.

        spacing

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder(spacing):
            args = protobufs.SpaceNullSamplerArgs()
            args.spacing = spacing
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "SpaceNull"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Space(
            col=input,
            extra={'type': 'RepeatNull',
                   'arg_builder': arg_builder,
                   'job_args': spacings})

    def Repeat(self,
               input: scannerpy.op.OpColumn,
               spacings) -> scannerpy.op.OpColumn:
        r"""Expands a sequence by repeating elements.

        Parameters
        ----------
        input
          The stream to expand.

        spacing

        Returns
        -------
        scannerpy.op.OpColumn
          The sampled stream.
        """
        def arg_builder(spacing):
            args = protobufs.SpaceRepeatSamplerArgs()
            args.spacing = spacing
            sampling_args = protobufs.SamplingArgs()
            sampling_args.sampling_function = "SpaceRepeat"
            sampling_args.sampling_args = args.SerializeToString()
            return sampling_args

        return self._db.ops.Space(
            col=input,
            extra={'type': 'Repeat',
                   'arg_builder': arg_builder,
                   'job_args': spacings})
