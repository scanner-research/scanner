from common import *
from column import Column


class Table:
    """
    A table in a Database.

    Can be part of many Collection objects.
    """
    def __init__(self, db, descriptor):
        self._db = db
        self._descriptor = descriptor
        job_id = self._descriptor.job_id
        if job_id != -1:
            self._job = self._db._load_descriptor(
                self._db._metadata_types.JobDescriptor,
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
        return columns[index] if index is not None else columns

    def num_rows(self):
        return self._descriptor.num_rows

    def rows(self):
        if self._job is None:
            return list(range(self.num_rows()))
        else:
            if len(self._task.samples) == 1:
                return list(self._task.samples[0].rows)
            else:
                return list(range(self.num_rows()))

    def profiler(self):
        job_id = self._descriptor.job_id
        if job_id != -1:
            return self._db.profiler(job_id)
        else:
            raise ScannerException('Ingested videos do not have profile data')
