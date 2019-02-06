class IOGenerator:
    def __init__(self, db):
        self._db = db

    def Input(self, storage):
        example = storage[0]
        source = example.source(self._db, storage)
        source._storage = storage
        return source

    def Output(self, op, storage):
        example = storage[0]
        sink = example.sink(self._db, op, storage)
        sink._storage = storage
        return sink
