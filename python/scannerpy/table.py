from common import *
from column import Column
import struct
from itertools import izip
from sampler import TableSampler

class Table:
    """
    A table in a Database.

    Can be part of many Collection objects.
    """
    def __init__(self, db, descriptor):
        self._db = db
        self._descriptor = descriptor
        self._collection = None
        job_id = self._descriptor.job_id
        if job_id != -1:
            self._job = self._db._load_descriptor(
                self._db.protobufs.JobDescriptor,
                'jobs/{}/descriptor.bin'.format(job_id))
            self._task = None
            for task in self._job.tasks:
                if task.output_table_name == self._descriptor.name:
                    self._task = task
            if self._task is None:
                raise ScannerException('Table {} not found in job {}'
                                       .format(self._descriptor.name, job_id))
        else:
            self._job = None

    def id(self):
        return self._descriptor.id

    def name(self):
        return self._descriptor.name

    def columns(self, index=None):
        columns = [Column(self, c) for c in self._descriptor.columns]
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
        return TableSampler(self)

    def num_rows(self):
        return self._descriptor.end_rows[-1]

    def _parse_index(self, bufs, db):
        return struct.unpack("=Q", bufs[0])[0]

    def parent_rows(self):
        if self._job is None:
            raise ScannerException('Table {} has no parent'.format(self.name()))

        return [i for _, i in self.load(['index'], fn=self._parse_index)]

    def profiler(self):
        job_id = self._descriptor.job_id
        if job_id != -1:
            return self._db.profiler(job_id)
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

    def frames(self):
        return Sampler()
