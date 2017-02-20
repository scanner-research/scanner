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

    def all(self, videos):
        return self.strided(videos, 1)

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

    def gather(self, video, rows):
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
        sample.rows.extend(rows)
        return task

    def strided_range(self, video, start, end, stride):
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
        sample.rows.extend(
            range(min(start, num_rows), min(end, num_rows), stride))
        return task
