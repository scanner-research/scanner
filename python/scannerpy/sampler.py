from common import *
from collection import Collection


class Sampler:
    """
    Utility for specifying which frames of a video (or which rows of a table)
    to run a computation over.
    """

    def __init__(self, db):
        self._db = db

    def _convert_collection(self, videos):
        if isinstance(videos, Collection):
            return [(t, '') for t in videos.table_names()]
        else:
            return videos

    def all(self, videos, item_size=1000, warmup_size=0):
        sampler_args = self._db.protobufs.AllSamplerArgs()
        sampler_args.sample_size = item_size
        sampler_args.warmup_size = warmup_size
        videos = self._convert_collection(videos)
        tasks = []
        for video in videos:
            (input_table_name, output_table_name) = video
            table = self._db.table(video[0])
            task = self._db.protobufs.Task()
            task.output_table_name = output_table_name
            input_table = self._db.table(input_table_name)
            column_names = [c.name() for c in input_table.columns()]
            sample = task.samples.add()
            sample.table_name = input_table_name
            sample.column_names.extend(column_names)
            sample.sampling_function = "All"
            sample.sampling_args = sampler_args.SerializeToString()
            tasks.append(task)
        return tasks

    def strided(self, videos, stride):
        videos = self._convert_collection(videos)
        tasks = []
        for video in videos:
            table = self._db.table(video[0])
            task = self.strided_range(video, 0, table.num_rows(), stride)
            tasks.append(task)
        return tasks

    def range(self, videos, start, end):
        videos = self._convert_collection(videos)
        tasks = []
        for video in videos:
            task = self.strided_range(video, start, end, 1)
            tasks.append(task)
        return tasks

    def gather(self, video, rows, item_size=1000):
        if isinstance(video, list) or isinstance(video, Collection):
            raise ScannerException('Sampler.gather only takes a single video')
        if not isinstance(video, tuple):
            raise ScannerException("""Error: sampler input must either be a collection \
or (input_table, output_table) pair')""")

        (input_table_name, output_table_name) = video
        task = self._db.protobufs.Task()
        task.output_table_name = output_table_name
        input_table = self._db.table(input_table_name)
        column_names = [c.name() for c in input_table.columns()]
        sample = task.samples.add()
        sample.table_name = input_table_name
        sample.column_names.extend(column_names)
        sample.sampling_function = "Gather"
        sampler_args = self._db.protobufs.GatherSamplerArgs()
        s = 0
        while s < len(rows):
            e = min(s + item_size, len(rows))
            sampler_args_sample = sampler_args.samples.add()
            sampler_args_sample.rows[:] = rows[s:e]
            s = e
        sample.sampling_args = sampler_args.SerializeToString()
        return task

    def strided_range(self, video, start, end, stride, item_size=1000,
                      warmup_size=0):
        if isinstance(video, list) or isinstance(video, Collection):
            raise ScannerException('Sampler.strided_range only takes a single video')
        if not isinstance(video, tuple):
            raise ScannerException("""Error: sampler input must either be a collection \
or (input_table, output_table) pair')""")

        (input_table_name, output_table_name) = video
        task = self._db.protobufs.Task()
        task.output_table_name = output_table_name
        input_table = self._db.table(input_table_name)
        num_rows = input_table.num_rows()
        column_names = [c.name() for c in input_table.columns()]
        sample = task.samples.add()
        sample.table_name = input_table_name
        sample.column_names.extend(column_names)
        sample.sampling_function = "StridedRange"
        sampler_args = self._db.protobufs.StridedRangeSamplerArgs()
        sampler_args.stride = stride
        s = start
        while s < end:
            ws = max(0, s - warmup_size * stride)
            e = min(s + item_size * stride, end)
            sampler_args.warmup_starts.append(ws)
            sampler_args.starts.append(s)
            sampler_args.ends.append(e)
            s = e
        sample.sampling_args = sampler_args.SerializeToString()
        return task
