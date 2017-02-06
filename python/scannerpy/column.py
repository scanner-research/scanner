import struct
import cv2
from common import *

class Column:
    """
    A column of a Table.
    """

    def __init__(self, table, descriptor):
        self._table = table
        self._descriptor = descriptor
        self._db = table._db
        self._storage = table._db.config.storage
        self._db_path = table._db.config.db_path

    def name(self):
        return self._descriptor.name

    def _load_output_file(self, item_id, rows, fn=None):
        assert len(rows) > 0

        path = '{}/tables/{}/{}_{}.bin'.format(
            self._db_path, self._table._descriptor.id,
            self._descriptor.id, item_id)
        try:
            contents = self._storage.read(path)
        except UserWarning:
            raise ScannerException('Path {} does not exist'.format(path))

        lens = []
        start_pos = None
        pos = 0
        (num_rows,) = struct.unpack("l", contents[:8])

        i = 8
        rows = rows if len(rows) > 0 else range(num_rows)
        for fi in rows:
            (buf_len,) = struct.unpack("l", contents[i:i+8])
            i += 8
            old_pos = pos
            pos += buf_len
            if start_pos is None:
                start_pos = old_pos
            lens.append(buf_len)

        i = 8 + num_rows * 8 + start_pos
        for buf_len in lens:
            buf = contents[i:i+buf_len]
            i += buf_len
            if fn is not None:
                yield fn(buf)
            else:
                yield buf

    def _load_all(self, fn=None):
        table_descriptor = self._table._descriptor
        total_rows = table_descriptor.num_rows
        rows_per_item = table_descriptor.rows_per_item

        # Integer divide, round up
        num_items = int(math.ceil(total_rows / float(rows_per_item)))
        bufs = []
        input_rows = self._table.rows()
        assert len(input_rows) == total_rows
        i = 0
        for item_id in range(num_items):
            rows = total_rows % rows_per_item \
                   if item_id == num_items - 1 else rows_per_item
            for output in self._load_output_file(item_id, range(rows), fn):
                yield (input_rows[i], output)
                i += 1

    def _decode_png(self, png):
        return cv2.imdecode(np.frombuffer(png, dtype=np.dtype(np.uint8)),
                            cv2.IMREAD_COLOR)

    def load(self, fn=None):
        """
        Loads the results of a Scanner computation into Python.

        Kwargs:
            fn: Optional function to apply to the binary blobs as they are read
                in.

        Returns:
            Generator that yields either a numpy array for frame columns or
            a binary blob for non-frame columns (optionally processed by the
            `fn`).
        """

        # If the column is a video, then dump the requested frames to disk as PNGs
        # and return the decoded PNGs
        if self._descriptor.type == self._db._metadata_types.Video:
            sampler = self._db.sampler()
            tasks = sampler.all([(self._table.name(), '__scanner_png_dump')])
            [out_tbl] = self._db.run(tasks, self._db.evaluators.ImageEncoder(),
                                     force=True)
            return out_tbl.columns(0).load(self._decode_png)
        else:
            return self._load_all(fn)
