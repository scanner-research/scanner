from common import *


class Collection:
    """
    A set of Table objects.
    """

    def __init__(self, db, name, descriptor):
        self._db = db
        self._name = name
        self._descriptor = descriptor
        self._tables = None

    def name(self):
        return self._name

    def table_names(self):
        return list(self._descriptor.tables)

    def tables(self, index=None):
        if self._tables is None:
            self._tables = [self._db.table(t) for t in self._descriptor.tables]
        return self._tables[index] if index is not None else self._tables

    def profiler(self):
        return self._db.profiler(self._descriptor.job_id)
