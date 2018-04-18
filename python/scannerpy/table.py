
import struct

from timeit import default_timer as now

from scannerpy.common import *
from scannerpy.column import Column


class Table:
    """
    A table in a Database.

    Can be part of many Collection objects.
    """

    def __init__(self, db, name, id):
        self._db = db
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
            self._descriptor = self._db._load_table_metadata([self._name])[0]

    def _load_column(self, name):
        if not self.committed():
            raise ScannerException('Table has not committed yet.')
        self._need_descriptor()
        if self._video_descriptors is None:
            self._video_descriptors = []
            for c in self._descriptor.columns:
                video_descriptor = None
                if c.type == self._db.protobufs.Video:
                    video_descriptor = self._db._load_descriptor(
                        self._db.protobufs.VideoDescriptor,
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
            self._job = self._db._load_descriptor(
                self._db.protobufs.JobDescriptor,
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
        return self._descriptor.end_rows[-1]

    def _parse_index(self, bufs, db):
        return struct.unpack("=Q", bufs[0])[0]

    def committed(self):
        return self._db._table_committed[self._id]

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
            return self._db.profiler(self._descriptor.job_id)
        else:
            raise ScannerException('Ingested videos do not have profile data')

    def load(self, columns, fn=None, rows=None):
        if not self.committed():
            raise ScannerException('Table has not committed yet.')
        cols = [self.column(c).load(rows=rows) for c in columns]
        for tup in zip(*cols):
            if fn is not None:
                yield fn(tup, self._db)
            else:
                yield tup
