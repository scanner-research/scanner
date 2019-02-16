from scannerpy.protobufs import protobufs
import scannerpy.types as scannertypes
from scannerpy.common import ScannerException
import os
import pickle


class NullElement:
    pass


class Storage:
    r"""I/O backend for streams fed in/out of Scanner."""

    def source(self, db, streams):
        raise ScannerException(
            "Storage class `{}` cannot serve as a Scanner input.".format(type(self).__name__))

    def sink(self, db, op, streams):
        raise ScannerException(
            "Storage class `{}` cannot serve as a Scanner output.".format(type(self).__name__))

    def delete(self, db, streams):
        raise ScannerException(
            "Storage class `{}` cannot delete elements.".format(type(self).__name__))


class StoredStream:
    def load_bytes(self, rows=None):
        raise ScannerException(
            "Stream `{}` cannot load elements into Python.".format(type(self).__name__))

    def committed(self) -> bool:
        raise NotImplementedError

    def exists(self) -> bool:
        raise NotImplementedError

    def storage(self) -> Storage:
        raise NotImplementedError

    def len(self) -> int:
        raise NotImplementedError

    def type(self):
        raise NotImplementedError

    def load(self, ty=None, fn=None, rows=None):
        if not self.committed():
            raise ScannerException("Tried to load from uncommitted stream")

        # Use a deserializer function if provided.
        # If not, use a type if provided.
        # If not, attempt to determine the type from the column's table descriptor.
        # If that doesn't work, then assume no deserialization function, and return bytes.
        if fn is None:
            if ty is None:
                ty = self.type()
            if ty is not None:
                fn = ty.deserializer

        for obj in self.load_bytes(rows=rows):
            if fn is not None and type(obj) == bytes:
                yield fn(obj)
            else:
                yield obj


class ScannerStorage(Storage):
    def source(self, db, streams):
        return db.sources.Column(
            table_name=[s._name for s in streams],
            column_name=['column' for s in streams])

    def sink(self, db, op, streams):
        return db.sinks.Column(
            columns={'column': op},
            table_name=[s._name for s in streams],
            column_name=['column' for s in streams])

    def delete(self, db, streams):
        if len(streams) > 0:
            db.delete_tables([e._name for e in streams])


class ScannerFrameStorage(ScannerStorage):
    def source(self, db, streams):
        return db.sources.FrameColumn(
            table_name=[s._name for s in streams],
            column_name=['frame' for s in streams])

    def sink(self, db, op, streams):
        return db.sinks.FrameColumn(
            columns={'frame': op},
            table_name=[s._name for s in streams],
            column_name=['frame' for s in streams])


class ScannerStream(StoredStream):
    def __init__(self, db, name, storage=None):
        if storage is None:
            self._storage = ScannerStorage()
        else:
            self._storage = storage

        self._db = db
        self._name = name

    def type(self):
        seq = self._db.sequence(self._name)
        seq._load_meta()
        type_name = seq._descriptor.type_name
        if type_name != "":
            return scannertypes.get_type_info_cpp(type_name)
        else:
            return None

    def storage(self):
        return self._storage

    def committed(self):
        return self._db.sequence(self._name)._table.committed()

    def exists(self):
        return self._db.sequence(self._name)._table.exists()

    def len(self):
        return self._db.sequence(self._name)._table.num_rows()

    def load_bytes(self, rows=None):
        seq = self._db.sequence(self._name)
        yield from seq.load(fn=lambda x: x, workers=16, rows=rows)


class ScannerFrameStream(ScannerStream):
    def __init__(self, db, name, storage=None):
        if storage is None:
            self._storage = ScannerFrameStorage()
        else:
            self._storage = storage

        self._db = db
        self._name = name

    def save_mp4(self, *args, **kwargs):
        return self._db.sequence(self._name).save_mp4(*args, **kwargs)


class FilesStorage(Storage):
    def __init__(self, storage_type="posix", bucket=None, region=None, endpoint=None):
        self._storage_type = storage_type
        self._bucket = bucket
        self._region = region
        self._endpoint = endpoint

    def source(self, db, streams):
        return db.sources.Files(
            storage_type=self._storage_type,
            bucket=self._bucket,
            region=self._region,
            endpoint=self._endpoint,
            paths=[s._paths for s in streams])

    def sink(self, db, op, streams):
        return db.sinks.Files(
            input=op,
            storage_type=self._storage_type,
            bucket=self._bucket,
            region=self._region,
            endpoint=self._endpoint,
            paths=[s._paths for s in streams])

    def delete(self, db, streams):
        # TODO
        pass


class FilesStream(StoredStream):
    def __init__(self, paths, storage=None):
        if storage is None:
            self._storage = FilesStorage()
        else:
            self._storage = storage

        self._paths = paths

    def load_bytes(self, rows=None):
        paths = self._paths
        if rows is not None:
            paths = [paths[i] for i in rows]

        for path in paths:
            yield open(path, 'rb').read()

    def storage(self):
        return self._storage

    def committed(self):
        # TODO
        return all(os.path.isfile(p) for p in self._paths)

    def exists(self):
        # TODO
        return any(os.path.isfile(p) for p in self._paths)

    def type(self):
        return None


class PythonStorage(Storage):
    def source(self, db, streams):
        return db.sources.Python(data=[pickle.dumps(stream._data) for stream in streams])


class PythonStream(StoredStream):
    def __init__(self, data):
        self._data = data

    def storage(self):
        return PythonStorage()


class SQLStorage(Storage):
    def __init__(self, config, job_table):
        self._config = config
        self._job_table = job_table

    def source(self, db, streams):
        num_elements = [s._num_elements for s in streams] \
                       if streams[0]._num_elements is not None else None
        return db.sources.SQL(
            query=streams[0]._query,
            config=self._config,
            enum_config=[self._config for _ in streams],
            enum_query=[s._query for s in streams],
            filter=[s._filter for s in streams],
            num_elements=num_elements)

    def sink(self, db, op, streams):
        return db.sinks.SQL(
            input=op,
            config=self._config,
            table=streams[0]._table,
            job_table=self._job_table,
            job_name=[s._job_name for s in streams],
            insert=streams[0]._insert)

    def delete(self, db, streams):
        # TODO
        pass


class SQLInputStream(StoredStream):
    def __init__(self, query, filter, storage, num_elements=None):
        assert isinstance(storage, SQLStorage)
        self._query = query
        self._storage = storage
        self._filter = filter
        self._num_elements = num_elements

    def storage(self):
        return self._storage


class SQLOutputStream(StoredStream):
    def __init__(self, table, job_name, storage, insert=True):
        assert isinstance(storage, SQLStorage)
        self._storage = storage
        self._table = table
        self._job_name = job_name
        self._insert = insert

    def storage(self):
        return self._storage

    def exists(self):
        # TODO
        return False

    def committed(self):
        # TODO
        return False


class AudioStorage(Storage):
    def source(self, db, streams):
        return db.sources.Audio(
            frame_size=[s._frame_size for s in streams],
            path=[s._path for s in streams])


class AudioStream(StoredStream):
    def __init__(self, path, frame_size, storage=None):
        if storage is None:
            self._storage = AudioStorage()
        else:
            self._storage = storage

        self._path = path
        self._frame_size = frame_size

    def storage(self):
        return self._storage


class CaptionStorage(Storage):
    def source(self, db, streams):
        return db.sources.Captions(
            window_size=[s._window_size for s in streams],
            path=[s._path for s in streams],
            max_time=[s._max_time for s in streams])


class CaptionStream(StoredStream):
    def __init__(self, path, window_size, max_time, storage=None):
        if storage is None:
            self._storage = CaptionStorage()
        else:
            self._storage = storage

        self._path = path
        self._max_time = max_time
        self._window_size = window_size

    def storage(self):
        return self._storage
