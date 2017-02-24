import struct
import cv2
import math
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

        descriptor_id = 1 - self._descriptor.id
        path = '{}/tables/{}/{}_{}.bin'.format(
            self._db_path, self._table._descriptor.id,
            descriptor_id, item_id)
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
        for fi in range(num_rows):
            (buf_len,) = struct.unpack("l", contents[i:i+8])
            i += 8
            old_pos = pos
            pos += buf_len
            if start_pos is None:
                start_pos = old_pos
            lens.append(buf_len)

        rows_idx = 0
        i = 8 + num_rows * 8 + start_pos
        for j, buf_len in enumerate(lens):
            if j == rows[rows_idx]:
                buf = contents[i:i+buf_len]
                if fn is not None:
                    return fn(buf, self._db)
                else:
                    return buf
                rows_idx += 1
            i += buf_len

    def _load(self, fn=None, rows=None):
        table_descriptor = self._table._descriptor
        total_rows = table_descriptor.end_rows[-1]

        # Integer divide, round up
        num_items = len(table_descriptor.end_rows)
        bufs = []
        input_rows = self._table.rows()
        assert len(input_rows) == total_rows
        i = 0
        rows_so_far = 0
        rows_idx = 0
        rows = range(total_rows) if rows is None else rows
        prev = 0
        for item_id in range(num_items):
            start_row = prev
            end_row = table_descriptor.end_rows[item_id]
            item_rows = start_row - end_row
            prev = end_row
            select_rows = []
            while rows_idx < len(rows):
                r = rows[rows_idx]
                if r >= start_row and r < end_row:
                    select_rows.append(r - start_row)
                    rows_idx += 1
                else:
                    break
            if select_rows:
                for output in self._load_output_file(item_id, select_rows, fn):
                    return (input_rows[i], output)
                    i += 1
            rows_so_far += item_rows

    def _decode_png(self, png):
        return cv2.imdecode(np.frombuffer(png, dtype=np.dtype(np.uint8)),
                            cv2.IMREAD_COLOR)

    def load(self, fn=None, rows=None):
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

        # If the column is a video, then dump the requested frames to disk as
        # PNGs and return the decoded PNGs
        if self._descriptor.type == self._db.protobufs.Video:
            sampler = self._db.sampler()
            png_table_name = self._db._png_dump_prefix.format(self._table.name())
            if self._db.has_table(png_table_name):
                png_table = self._db.table(png_table_name)
                if png_table._descriptor.timestamp > \
                   self._table._descriptor.timestamp:
                    return png_table.columns(0).load(self._decode_png)
            pair = [(self._table.name(), png_table_name)]
            if rows is None:
                tasks = sampler.all(pair)
            else:
                tasks = [sampler.gather(pair, rows)]
            [out_tbl] = self._db.run(tasks, self._db.ops.ImageEncoder(),
                                     force=True)
            return out_tbl.columns(0).load(self._decode_png)
        else:
            return self._load(fn, rows=rows)
