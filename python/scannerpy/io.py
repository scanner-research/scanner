from scannerpy.common import ScannerException
from scannerpy.storage import StoredStream

class IOGenerator:
    def __init__(self, sc):
        self._sc = sc

    def Input(self, streams):
        if not isinstance(streams, list) or not isinstance(streams[0], StoredStream):
            raise ScannerException("io.Input must take a list of streams as input")

        example = streams[0]
        source = example.storage().source(self._sc, streams)
        source._streams = streams
        return source

    def Output(self, op, streams):
        if not isinstance(streams, list) or not isinstance(streams[0], StoredStream):
            raise ScannerException("io.Output must take a list of streams as input")

        example = streams[0]
        sink = example.storage().sink(self._sc, op, streams)
        sink._streams = streams
        return sink
