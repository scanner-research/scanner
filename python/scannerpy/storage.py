from scannerpy.protobufs import protobufs
from scannerpy.common import ScannerException
import os
import pickle
from typing import List, Callable, Any, Generator
from scannerpy.types import get_type_info_cpp

class NullElement:
    """Represents a 'null' output in a stream generated through Scanner.

    These can show up e.g. when you use a spacing/repeat operator. NullElement is used instead
    of the more generic Python 'None', as a Python kernel can have 'None' as a valid output which
    isn't the same as a 'null' stream element.
    """

    pass


class Storage:
    """I/O backend for streams fed in/out of Scanner."""

    def source(self, sc, streams):
        """Returns the Scanner source corresponding to this storage backend.

        Parameters
        ----------
        sc: Client
          Scanner client

        streams: List[StoredStream]
          List of StoredStream objects of the same storage type

        Returns
        ----------
        source: Source
          Scanner source
        """

        raise ScannerException(
            "Storage class `{}` cannot serve as a Scanner input.".format(type(self).__name__))

    def sink(self, sc, op, streams):
        """Returns the Scanner sink corresponding to this storage backend.

        Parameters
        ----------
        sc: Client
          Scanner client

        op: Op
          Input op from computation graph

        streams: List[StoredStream]
          List of StoredStream objects of the same storage type

        Returns
        ----------
        sink: Sink
          Scanner sink
        """

        raise ScannerException(
            "Storage class `{}` cannot serve as a Scanner output.".format(type(self).__name__))

    def delete(self, sc, streams):
        """Deletes the streams from storage if they exist.

        Parameters
        ----------
        sc: Client
          Scanner client

        streams: List[StoredStream]
          List of StoredStream objects of the same storage type
        """

        raise ScannerException(
            "Storage class `{}` cannot delete elements.".format(type(self).__name__))


class StoredStream:
    """Handle to a stream stored in a particular storage backend.

    In general, a StoredStream is not guaranteed to exist, but you can always check this using
    :func:`exists`.
    """

    def load_bytes(self, rows: List[int] = None) -> Generator[bytes, None, None]:
        """A generator that incrementally loads raw bytes from the stored stream.

        This function is not intended to be called directly by the user. See :func:`load`.

        Parameters
        ----------
        rows: List[int]
          List of indices in the stream to load. Default is all elements.
        """

        raise ScannerException(
            "Stream `{}` cannot load elements into Python.".format(type(self).__name__))

    def committed(self) -> bool:
        """Check if a stream is completely materialized in storage.

        A stream may exist, but not be committed if a Scanner job was run that was supposed to
        output the stream, but the job failed for any reason.
        """
        raise NotImplementedError

    def exists(self) -> bool:
        """Check if any part of a stream exists in storage."""
        raise NotImplementedError

    def storage(self) -> Storage:
        """Get the storage backend corresponding to this stream."""
        raise NotImplementedError

    def len(self) -> int:
        """Get the number of elements in this stream."""
        raise NotImplementedError

    def type(self) -> type:
        """Get the Scanner type of elements in this stream if it exists, and return None otherwise."""
        raise NotImplementedError

    def load(self, ty: type = None, fn: Callable[[bytes], Any] = None, rows: List[int] = None) -> Generator[Any, None, None]:
        """Load elements from the stored stream into Python.

        Parameters
        ----------
        ty: type
          Scanner type of elements in the stream. If the storage backend recorded the type, that
          will be used by default.

        fn: Callable[[bytes], Any]
          Deserialization function that maps elements as bytes to their actual type.

        rows: List[int]
          List of indices in the stream to load. Default is all elements.

        Returns
        ----------
        generator: Generator[Any, None, None]
          A generator that outputs one stream element at a time.
        """

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

    def delete(self, sc):
        """Deletes the stream from its storage if it exists.

        Parameters
        ----------
        sc: Client
          Scanner client
        """

        self.storage().delete(sc, [self])


class NamedStorage(Storage):
    """Named storage for byte streams. Useful default output format for non-video-data.

    Stores byte streams in a custom packed binary file format. Supports both local filesystems
    (Linux/Posix) and cloud file systems (S3, GCS).
    """

    def source(self, sc, streams):
        return sc.sources.Column(
            table_name=[s._name for s in streams],
            column_name=['column' for s in streams])

    def sink(self, sc, op, streams):
        return sc.sinks.Column(
            columns={'column': op},
            table_name=[s._name for s in streams],
            column_name=['column' for s in streams])

    def delete(self, sc, streams):
        if len(streams) > 0:
            sc.delete_tables([e._name for e in streams])


class NamedVideoStorage(NamedStorage):
    """Named storage for video streams. Special baked-in storage class for keeping videos compressed."""

    def source(self, sc, streams):
        return sc.sources.FrameColumn(
            table_name=[s._name for s in streams],
            column_name=['frame' for s in streams])

    def sink(self, sc, op, streams):
        return sc.sinks.FrameColumn(
            columns={'frame': op},
            table_name=[s._name for s in streams],
            column_name=['frame' for s in streams])

    def ingest(self, sc, streams, batch=500):
        to_ingest = [(s._name, s._path) for s in streams
                     if (s._path is not None and
                         s._inplace == False)]
        to_ingest_inplace = [(s._name, s._path) for s in streams
                             if (s._path is not None and
                                 s._inplace == True)]
        if len(to_ingest) > 0:
            for i in range(0, len(to_ingest), batch):
                sc.ingest_videos(to_ingest[i:i+batch], inplace=False, force=True)
        if len(to_ingest_inplace) > 0:
            for i in range(0, len(to_ingest_inplace), batch):
                sc.ingest_videos(to_ingest_inplace[i:i+batch], inplace=True, force=True)


class NamedStream(StoredStream):
    """Stream of elements stored on disk with a given name."""

    def __init__(self, sc, name: str, storage=None):
        """
        Parameters
        ----------
        sc: Client
          Scanner client.

        name: str
          Name of the stream.

        storage: NamedStorage
          Optional NamedStorage object.
        """

        if storage is None:
            self._storage = NamedStorage()
        else:
            self._storage = storage

        self._sc = sc
        self._name = name

    def name(self):
        return self._name

    def type(self):
        seq = self._sc.sequence(self._name)
        seq._load_meta()
        type_name = seq._descriptor.type_name
        if type_name != "":
            return get_type_info_cpp(type_name)
        else:
            return None

    def storage(self):
        return self._storage

    def committed(self):
        return self._sc.sequence(self._name)._table.committed()

    def exists(self):
        return self._sc.sequence(self._name)._table.exists()

    def len(self):
        return self._sc.sequence(self._name)._table.num_rows()

    def load_bytes(self, rows=None):
        seq = self._sc.sequence(self._name)
        yield from seq.load(fn=lambda x: x, workers=16, rows=rows)


class NamedVideoStream(NamedStream):
    """Stream of video frame stored (compressed) in named storage."""

    def __init__(self, sc, name, path=None, inplace=False, storage=None):
        """
        Parameters
        ----------
        sc: Client
          Scanner client

        name: str
          Name of the stream in Scanner storage.

        path: str
          Path to corresponding video file. If this parameter is given, the named video stream will be
          created from the video at the given path.

        storage: NamedFrameStorage
          Optional NamedFrameStorage object.
        """

        if storage is None:
            self._storage = NamedVideoStorage()
        else:
            self._storage = storage

        self._sc = sc
        self._name = name
        self._path = path
        self._inplace = inplace

    def save_mp4(self, output_name, fps=None, scale=None):
        return self._sc.sequence(self._name).save_mp4(output_name, fps=fps, scale=scale)


class FilesStorage(Storage):
    """Storage of streams where each element is its own file."""

    def __init__(self, storage_type: str = "posix", bucket: str = None, region: str = None, endpoint: str = None):
        """
        Parameters
        ----------
        storage_type: str
          Kind of filesystem the files are on. Either "posix" or "gcs" supported.

        bucket: str
          If filesystem is gcs, name of bucket

        region: str
          If filesytem is gcs, region name of bucket, e.g. us-west1

        endpoint: str
          If filesystem is gcs, URL of storage endpoint
        """

        self._storage_type = storage_type
        self._bucket = bucket
        self._region = region
        self._endpoint = endpoint

    def source(self, sc, streams):
        return sc.sources.Files(
            storage_type=self._storage_type,
            bucket=self._bucket,
            region=self._region,
            endpoint=self._endpoint,
            paths=[s._paths for s in streams])

    def sink(self, sc, op, streams):
        return sc.sinks.Files(
            input=op,
            storage_type=self._storage_type,
            bucket=self._bucket,
            region=self._region,
            endpoint=self._endpoint,
            paths=[s._paths for s in streams])

    def delete(self, sc, streams):
        # TODO
        pass


class FilesStream(StoredStream):
    """Stream where each element is a file."""

    def __init__(self, paths: List[str], storage: FilesStorage = None):
        """
        Parameters
        ----------
        paths: List[str]
          List of paths to the files in the stream.

        storage: FilesStorage
        """
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
    """Storage for a stream of elements directly from the current Python process.

    Only supports input, not output.
    """

    def source(self, sc, streams):
        return sc.sources.Python(data=[pickle.dumps(stream._data) for stream in streams])


class PythonStream(StoredStream):
    """Stream of elements directly in the current Python process."""

    def __init__(self, data: List[Any]):
        """
        Parameters
        ----------
        data: List[Any]
          Arbitrary data to stream. Must be pickleable.
        """
        self._data = data

    def storage(self):
        return PythonStorage()


class SQLStorage(Storage):
    """Storage backend for streams from a SQL database.

    Currently only supports Postgres."""

    def __init__(self, config, job_table):
        """
        Parameters
        ----------
        config: protobufs.SQLConfig
          Database connection parameters

        job_table: str
          Name of table in the database to track completed jobs
        """

        self._config = config
        self._job_table = job_table

    def source(self, sc, streams):
        num_elements = [s._num_elements for s in streams] \
                       if streams[0]._num_elements is not None else None
        return sc.sources.SQL(
            query=streams[0]._query,
            config=self._config,
            enum_config=[self._config for _ in streams],
            enum_query=[s._query for s in streams],
            filter=[s._filter for s in streams],
            num_elements=num_elements)

    def sink(self, sc, op, streams):
        return sc.sinks.SQL(
            input=op,
            config=self._config,
            table=streams[0]._table,
            job_table=self._job_table,
            job_name=[s._job_name for s in streams],
            insert=streams[0]._insert)

    def delete(self, sc, streams):
        # TODO
        pass


class SQLInputStream(StoredStream):
    """Stream of elements from a SQL database used as input."""

    def __init__(self, query, filter, storage, num_elements=None):
        """
        Parameters
        ----------
        query: protobufs.SQLQuery
          Query that generates a table

        filter: str
          Filter on the query that picks the rows/elements only in this stream

        storage: SQLStorage

        num_elements: int
          Number of elements in this stream. Optional optimization to avoid Scanner having to count.
        """

        assert isinstance(storage, SQLStorage)
        self._query = query
        self._storage = storage
        self._filter = filter
        self._num_elements = num_elements

    def storage(self):
        return self._storage


class SQLOutputStream(StoredStream):
    """Stream of elements into a SQL database used as output."""

    def __init__(self, table, job_name, storage, insert=True):
        """
        Parameters
        ----------
        table: str
          Name of table to stream into.

        job_name: str
          Name of job to insert into the job table.

        storage: SQLStorage

        insert: bool
          Whether to insert new rows or update existing rows.
        """

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
    """Storage for stream of elements from a compressed audio file.

    Currently input-only."""

    def source(self, sc, streams):
        return sc.sources.Audio(
            frame_size=[s._frame_size for s in streams],
            path=[s._path for s in streams])


class AudioStream(StoredStream):
    """Stream of elements from a compressed audio file."""

    def __init__(self, path, frame_size, storage=None):
        """
        Parameters
        ----------
        path: str
          Path on filesystem to audio file.

        frame_size: float
          Size (in seconds) of each element, e.g. a 2s frame size with a a 44.1 kHz generates
          stream elements of 88.2k samples per element.

        storage: AudioStorage
        """

        if storage is None:
            self._storage = AudioStorage()
        else:
            self._storage = storage

        self._path = path
        self._frame_size = frame_size

    def storage(self):
        return self._storage


class CaptionStorage(Storage):
    """Storage for caption streams."""

    def source(self, sc, streams):
        return sc.sources.Captions(
            window_size=[s._window_size for s in streams],
            path=[s._path for s in streams],
            max_time=[s._max_time for s in streams])


class CaptionStream(StoredStream):
    """Stream of captions out of a caption file.

    In order for the number of stream elements to be predictable (e.g. to zip a caption stream
    with an audio stream for transcript alignment), we represent caption streams as uniform time
    intervals. You provide a stream duration (e.g. 10 minutes) and a window size (e.g. 5 seconds)
    and the stream contains 10 minutes / 5 seconds number of elements, where each element contains
    all of the text for captions that overlap with that window.
    """

    def __init__(self, path, window_size, max_time, storage=None):
        """
        Parameters
        ----------
        path: str
          Path on the filesystem to the caption file.

        window_size: float
          Size of window in time (seconds) for each element. See class description.

        max_time: float
          Total time for the entire stream. See class description.

        storage: CaptionStorage
        """

        if storage is None:
            self._storage = CaptionStorage()
        else:
            self._storage = storage

        self._path = path
        self._max_time = max_time
        self._window_size = window_size

    def storage(self):
        return self._storage
