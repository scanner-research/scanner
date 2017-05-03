from common import *
from column import Column
import struct
from itertools import izip
from sampler import SamplerOp
from timeit import default_timer as now

class Table:
    """
    A table in a Database.

    Can be part of many Collection objects.
    """
    def __init__(self, db, descriptor):
        self._db = db
        self._descriptor = descriptor
        self._columns = []
        self._job_id = self._descriptor.job_id

    def id(self):
        return self._descriptor.id

    def name(self):
        return self._descriptor.name

    def _load_columns(self):
        for c in self._descriptor.columns:
            video_descriptor = None
            if c.type == self._db.protobufs.Video:
                video_descriptor = self._db._load_descriptor(
                    self._db.protobufs.VideoDescriptor,
                    'tables/{:d}/{:d}_0_video_metadata.bin'.format(
                        self._descriptor.id,
                        c.id))
            self._columns.append(Column(self, c, video_descriptor))

    def _load_job(self):
        if self._job_id != -1:
            self._job = self._db._load_descriptor(
                self._db.protobufs.JobDescriptor,
                'jobs/{}/descriptor.bin'.format(self._job_id))
            self._task = None
            for task in self._job.tasks:
                if task.output_table_name == self._descriptor.name:
                    self._task = task
            if self._task is None:
                raise ScannerException('Table {} not found in job {}'
                                       .format(self._descriptor.name, job_id))
        else:
            self._job = None


    # HACK(wcrichto): reading from TableDescriptor to avoid loading VideoDescriptors
    def column_names(self):
        return [c.name for c in self._descriptor.columns]

    def column(self, index):
        return self.columns(index)

    def columns(self, index=None):
        if len(self._columns) == 0:
            self._load_columns()

        columns = self._columns
        if index is not None:
            col = None
            if isinstance(index, basestring):
                for c in columns:
                    if c.name() == index:
                        col = c
                        break
                if col is None:
                    raise ScannerException('Could not find column with name {}'
                                           .format(index))
            else:
                assert isinstance(index, int)
                if index < 0 or index >= len(columns):
                    raise ScannerException('No column with index {}'
                                           .format(index))
                col = columns[index]
            return col
        else:
            return columns

    def as_op(self):
        return SamplerOp(self)

    def num_rows(self):
        return self._descriptor.end_rows[-1]

    def _parse_index(self, bufs, db):
        return struct.unpack("=Q", bufs[0])[0]

    def parent_rows(self):
        if self._job_id == -1:
            raise ScannerException('Table {} has no parent'.format(self.name()))

        return [i for _, i in self.load(['index'], fn=self._parse_index)]

    def profiler(self):
        if self._job_id != -1:
            return self._db.profiler(self._job_id)
        else:
            raise ScannerException('Ingested videos do not have profile data')

    def load(self, columns, fn=None, rows=None):
        cols = [self.columns(c).load(rows=rows) for c in columns]
        for tup in izip(*cols):
            row = tup[0][0]
            vals = [x for _, x in tup]
            if fn is not None:
                yield (row, fn(vals, self._db))
            else:
                yield (row, vals)
