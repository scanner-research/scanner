from common import *

class TableSampler:
    """
    Utility for specifying which frames of a video (or which rows of a table)
    to run a computation over.
    """

    def __init__(self, table):
        self._table = table
        self._db = table._db

    def all(self, task_size=1000, warmup_size=0):
        sampler_args = self._db.protobufs.AllSamplerArgs()
        sampler_args.sample_size = task_size
        sampler_args.warmup_size = warmup_size
        task = self._db.protobufs.Task()
        #task.output_table_name = output_table_name
        column_names = [c.name() for c in self._table.columns()]
        sample = task.samples.add()
        sample.table_name = self._table.name()
        sample.column_names.extend(column_names)
        sample.sampling_function = "All"
        sample.sampling_args = sampler_args.SerializeToString()
        return task

    def strided(self, stride, task_size=1000):
        return self.strided_range(0, self._table.num_rows(), stride, task_size=task_size)

    def range(self, start, end, task_size=1000, warmup_size=0):
        return self.ranges([(start, end)], task_size=task_size,
                           warmup_size=warmup_size)

    def ranges(self, intervals, task_size=1000, warmup_size=0):
        return self.strided_ranges(
            intervals, 1,
            task_size=task_size,
            warmup_size=warmup_size)

    def gather(self, rows, task_size=1000):
        task = self._db.protobufs.Task()
        #task.output_table_name = output_table_name
        column_names = [c.name() for c in self._table.columns()]
        sample = task.samples.add()
        sample.table_name = self._table.name()
        sample.column_names.extend(column_names)
        sample.sampling_function = "Gather"
        sampler_args = self._db.protobufs.GatherSamplerArgs()
        s = 0
        while s < len(rows):
            e = min(s + task_size, len(rows))
            sampler_args_sample = sampler_args.samples.add()
            sampler_args_sample.rows[:] = rows[s:e]
            s = e
        sample.sampling_args = sampler_args.SerializeToString()
        return task

    def strided_range(self, start, end, stride, task_size=1000,
                      warmup_size=0):
        return self.strided_ranges([(start, end)], stride,
                                   task_size=task_size,
                                   warmup_size=warmup_size)

    def strided_ranges(self, intervals, stride, task_size=1000,
                      warmup_size=0):
        task = self._db.protobufs.Task()
        #task.output_table_name = output_table_name
        num_rows = self._table.num_rows()
        column_names = [c.name() for c in self._table.columns()]
        sample = task.samples.add()
        sample.table_name = self._table.name()
        sample.column_names.extend(column_names)
        sample.sampling_function = "StridedRange"
        sampler_args = self._db.protobufs.StridedRangeSamplerArgs()
        sampler_args.stride = stride
        for start, end in intervals:
            s = start
            while s < end:
                ws = max(0, s - warmup_size * stride)
                e = min(s + task_size * stride, end)
                sampler_args.warmup_starts.append(ws)
                sampler_args.starts.append(s)
                sampler_args.ends.append(e)
                s = e
        sample.sampling_args = sampler_args.SerializeToString()
        return task

class SamplerOp:
    def __init__(self, table):
        from collection import Collection
        if isinstance(table, Collection):
            self._collection = table
            self._table = self._collection.tables(0)
        else:
            self._collection = None
            self._table = table

    def __getattr__(self, attr):
        def fn(*args, **kwargs):
            def task_generator(t=self._table):
                return getattr(TableSampler(t), attr)(*args, **kwargs)
            return self._table._db.ops.Input(
                self._table.columns(),
                task_generator,
                self._collection).outputs()

        return fn
