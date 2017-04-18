import struct
import cv2
import math
from common import *
from stdlib import parsers
from subprocess import Popen, PIPE
import tempfile
import os

class Column:
    """
    A column of a Table.
    """

    def __init__(self, table, descriptor, video_descriptor):
        self._table = table
        self._descriptor = descriptor
        self._db = table._db
        self._storage = table._db.config.storage
        self._db_path = table._db.config.db_path
        self._video_descriptor = video_descriptor

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
                    yield fn(buf, self._db)
                else:
                    yield buf
                rows_idx += 1
            i += buf_len

    def _load(self, fn=None, rows=None):
        table_descriptor = self._table._descriptor
        total_rows = table_descriptor.end_rows[-1]

        # Integer divide, round up
        num_items = len(table_descriptor.end_rows)
        bufs = []
        input_rows = list(range(self._table.num_rows()))
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
                    yield (input_rows[i], output)
                    i += 1
            rows_so_far += item_rows

    # TODO(wcrichto): don't show progress bar when running decode png
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
        if (self._descriptor.type == self._db.protobufs.Video and
            self._video_descriptor.codec_type ==
            self._db.protobufs.VideoDescriptor.H264):
            png_table_name = self._db._png_dump_prefix.format(self._table.name())
            if self._db.has_table(png_table_name):
                png_table = self._db.table(png_table_name)
                if rows is None and \
                   png_table.num_rows() == self._table.num_rows() and \
                   png_table._descriptor.timestamp > \
                   self._table._descriptor.timestamp:
                    return png_table.load(['png'], parsers.image)
            pair = [(self._table.name(), png_table_name)]
            if rows is None:
                frame = self._table.as_op().all()
            else:
                frame = self._table.as_op().gather(rows)
            img = self._db.ops.ImageEncoder(frame = frame)
            job = Job(columns = [img], name = png_table_name)
            [out_tbl] = self._db.run([job], force=True, show_progress=False)
            return out_tbl.load(['png'], parsers.image)
        elif self._descriptor.type == self._db.protobufs.Video:
            frame_type = self._video_descriptor.frame_type
            if frame_type == self._db.protobufs.U8:
                dtype = np.u8
            elif frame_type == self._db.protobufs.F32:
                dtype = np.float32
            elif frame_type == self._db.protobufs.F64:
                dtype = np.float64
            parser_fn = parsers.raw_frame_gen(self._video_descriptor.height,
                                              self._video_descriptor.width,
                                              self._video_descriptor.channels,
                                              dtype)
            return self._load(fn=parser_fn, rows=rows)
        else:
            return self._load(fn, rows=rows)

    def save_mp4(self, path, fps=None, rows=None):
        if not (self._descriptor.type == self._db.protobufs.Video and
                self._video_descriptor.codec_type ==
                self._db.protobufs.VideoDescriptor.H264):
            raise ScannerException('Attempted to save a non-h264-compressed '
                                   'column as an mp4. Try compressing the '
                                   'column first by saving the output as '
                                   'an RGB24 frame')
        num_items = len(self._table._descriptor.end_rows)

        paths = ['{}/tables/{:d}/{:d}_{:d}.bin'.format(
            self._db._db_path,
            self._table._descriptor.id, self._descriptor.id, item_id)
                          for item_id in range(num_items)]
        temp_paths = []
        for _ in range(len(paths)):
            fd, p = tempfile.mkstemp()
            os.close(fd)
            temp_paths.append(p)
        # Copy all files locally before calling ffmpeg
        for in_path, temp_path in zip(paths, temp_paths):
            with open(temp_path, 'w') as f:
                f.write(self._storage.read(in_path))

        files = '|'.join(temp_paths)

        vid_fps = (fps or
                   (1.0/(self._video_descriptor.time_base_num /
                         float(self._video_descriptor.time_base_denom))))
        cmd = (
            'ffmpeg -y '
            '-r {:f} '
            '-i "concat:{:s}" '
            '-filter:v "setpts=N" '
            '-bsf:a aac_adtstoasc '
            '{:s}'.format(fps, files, path))
        Popen(cmd, shell=True).wait()
