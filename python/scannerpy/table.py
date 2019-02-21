
import struct

from timeit import default_timer as now

from scannerpy.common import *
from scannerpy.protobufs import protobufs
from scannerpy.column import Column


class Table:
    """
    A table in a Database.

    Can be part of many Collection objects.
    """

    def __init__(self, sc, name, id):
        self._sc = sc
        # We pass name and id to avoid having to read the descriptor
        self._name = name
        self._id = id
        self._descriptor = None
        self._video_descriptors = None

    def id(self):
        return self._id

    def name(self):
        return self._name

    def _need_descriptor(self):
        if self._descriptor is None:
            self._descriptor = self._sc._load_table_metadata([self._name])[0]

    def _load_column(self, name):
        if not self.committed():
            raise ScannerException('Table has not committed yet.')
        self._need_descriptor()
        if self._video_descriptors is None:
            self._video_descriptors = []
            for c in self._descriptor.columns:
                video_descriptor = None
                if c.type == protobufs.Video:
                    video_descriptor = self._sc._load_descriptor(
                        protobufs.VideoDescriptor,
                        'tables/{:d}/{:d}_0_video_metadata.bin'.format(
                            self._id, c.id))
                self._video_descriptors.append(video_descriptor)
        for i, c in enumerate(self._descriptor.columns):
            if c.name == name:
                return c, self._video_descriptors[i]
        raise ScannerException('Column {} not found in Table {}'.format(
            name, self._name))

    def _load_job(self):
        self._need_descriptor()
        if self._descriptor.job_id != -1:
            self._job = self._sc._load_descriptor(
                protobufs.JobDescriptor,
                'jobs/{}/descriptor.bin'.format(self._descriptor.job_id))
            self._task = None
            for task in self._job.tasks:
                if task.output_table_name == self._name:
                    self._task = task
            if self._task is None:
                raise ScannerException('Table {} not found in job {}'.format(
                    self._name, self._descriptor.job_id))
        else:
            self._job = None

    # HACK(wcrichto): reading from TableDescriptor to avoid loading VideoDescriptors
    def column_names(self):
        self._need_descriptor()
        return [c.name for c in self._descriptor.columns]

    def column(self, name):
        return Column(self, name)

    def num_rows(self):
        self._need_descriptor()
        if len(self._descriptor.end_rows) > 0:
            return self._descriptor.end_rows[-1]
        else:
            return 0

    def _parse_index(self, bufs, sc):
        return struct.unpack("=Q", bufs[0])[0]

    def committed(self):
        return self.exists() and self._sc._table_committed[self._id]

    def exists(self):
        return self._id in self._sc._table_committed

    def parent_rows(self):
        self._need_descriptor()
        if self._descriptor.job_id == -1:
            raise ScannerException('Table {} has no parent'.format(
                self.name()))

        return [i for _, i in self.load(['index'], fn=self._parse_index)]

    def profiler(self):
        if not self.committed():
            raise ScannerException('Table has not committed yet.')
        self._need_descriptor()
        if self._descriptor.job_id != -1:
            return self._sc.profiler(self._descriptor.job_id)
        else:
            raise ScannerException('Ingested videos do not have profile data')

    def load(self, columns, fn=None, rows=None):
        if not self.committed():
            raise ScannerException('Table has not committed yet.')
        cols = [self.column(c).load(rows=rows) for c in columns]
        for tup in zip(*cols):
            if fn is not None:
                yield fn(tup, self._sc)
            else:
                yield tup
