from scannerpy.protobufs import protobufs

class Storage:
    def type(self):
        pass

    def load_bytes(self):
        raise NotImplemented

    def load(self, ty=None, fn=None):
        # Use a deserializer function if provided.
        # If not, use a type if provided.
        # If not, attempt to determine the type from the column's table descriptor.
        # If that doesn't work, then assume no deserialization function, and return bytes.
        if fn is None:
            if ty is None:
                ty = self.type()
            if ty is not None:
                fn = ty.deserializer

        for obj in self.load_bytes():
            if fn is not None:
                yield fn(obj)
            else:
                yield obj


class ScannerStorage(Storage):
    def __init__(self, seq):
        self._seq = seq

    def type(self):
        self._seq._load_meta()
        type_name = self._seq._descriptor.type_name
        if type_name != "":
            return scannertypes.get_type_info_cpp(type_name)

    def source(self, db, storage):
        self._seq._load_meta()
        kwargs = dict(
            table_name=[s._seq._table.name() for s in storage],
            column_name=[s._seq.name() for s in storage])
        if (self._seq._descriptor.type == protobufs.Video):
            return db.sources.FrameColumn(**kwargs)
        else:
            return db.sources.Column(**kwargs)

    def sink(self, db, op, storage):
        self._seq._load_meta()
        kwargs = dict(
            table_name=[s._seq._table.name() for s in storage],
            column_name=[s._seq.name() for s in storage])
        if (self._seq._descriptor.type == protobufs.Video):
            return db.sinks.FrameColumn(columns={'column': op}, **kwargs)
        else:
            return db.sinks.Column(columns={'column': op}, **kwargs)

class FileStorage(Storage):
    def __init__(self, paths):
        self._paths = paths

    def load_bytes(self):
        for path in self._paths:
            yield open(path, 'rb').read()

    def source(self, db, storage):
        return db.sources.Files(paths=[s._paths for s in storage])

    def sink(self, db, op, storage):
        return db.sinks.Files(input=op, paths=[s._paths for s in storage])
