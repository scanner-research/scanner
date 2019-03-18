from scannerpy.common import ScannerException
from typing import List, Callable, Any, Generator
from scannerpy.types import get_type_info_cpp


class NullElement:
    """Represents a 'null' output in a stream generated through Scanner.

    These can show up e.g. when you use a spacing/repeat operator. NullElement is used instead
    of the more generic Python 'None', as a Python kernel can have 'None' as a valid output which
    isn't the same as a 'null' stream element.
    """

    pass


class StorageBackend:
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
            "StorageBackend class `{}` cannot serve as a Scanner input.".format(type(self).__name__))

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
            "StorageBackend class `{}` cannot serve as a Scanner output.".format(type(self).__name__))

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
            "StorageBackend class `{}` cannot delete elements.".format(type(self).__name__))


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

    def storage(self) -> StorageBackend:
        """Get the storage backend corresponding to this stream."""
        raise NotImplementedError

    def len(self) -> int:
        """Get the number of elements in this stream."""
        raise NotImplementedError

    def type(self) -> type:
        """Get the Scanner type of elements in this stream if it exists, and return None otherwise."""
        raise NotImplementedError

    def estimate_size(self) -> int:
        """Estimates the size in bytes of elements in the stream. Not guaranteed to be accurate,
        just used for heuristics."""
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


class NamedStorage(StorageBackend):
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

    def estimate_size(self):
        col = self._sc.sequence(self._name)
        col._load_meta()
        vmeta = col._video_descriptor
        return vmeta.width * vmeta.height * vmeta.channels

    def save_mp4(self, output_name, fps=None, scale=None):
        return self._sc.sequence(self._name).save_mp4(output_name, fps=fps, scale=scale)
