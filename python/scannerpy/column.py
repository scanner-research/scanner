import struct
import math
from subprocess import Popen, PIPE
import tempfile
import os

from storehouse import RandomReadFile
from scannerpy.stdlib import readers
from scannerpy.common import *
from scannerpy.job import Job

LOAD_SPARSITY_THRESHOLD = 10


class Column(object):
    """
    A column of a Table.
    """

    def __init__(self, table, name):
        self._table = table
        self._name = name
        self._db = table._db
        self._storage = table._db.config.storage
        self._db_path = table._db.config.db_path

        self._loaded = False
        self._descriptor = None
        self._video_descriptor = None

    def _load_meta(self):
        if not self._loaded:
            self._loaded = True
            descriptor, video_descriptor = self._table._load_column(self._name)
            self._descriptor = descriptor
            self._video_descriptor = video_descriptor

    def name(self):
        return self._name

    def type(self):
        self._load_meta()
        return self._descriptor.type

    def id(self):
        self._load_meta()
        return self._descriptor.id

    def keyframes(self):
        self._load_meta()
        if (self._descriptor.type == self._db.protobufs.Video
                and self._video_descriptor.codec_type ==
                self._db.protobufs.VideoDescriptor.H264):
            # For each encoded video, add start frame offset
            frame_offset = 0
            kf_offset = 0
            keyframes = []
            for frames_per_video, kfs_per_video in zip(
                    self._video_descriptor.frames_per_video,
                    self._video_descriptor.keyframes_per_video):
                keyframes += [
                    frame_offset + kfi
                    for kfi in self._video_descriptor.keyframe_indices[
                        kf_offset:kf_offset + kfs_per_video]
                ]
                frame_offset += frames_per_video
                kf_offset += kfs_per_video
            return keyframes
        else:
            return list(range(self._table.num_rows()))

    def _load_output_file(self, item_id, rows, fn=None):
        assert len(rows) > 0

        metadata_path = '{}/tables/{}/{}_{}_metadata.bin'.format(
            self._db_path, self._table._descriptor.id, self._descriptor.id,
            item_id)
        try:
            metadata_file = RandomReadFile(self._storage, metadata_path)
        except UserWarning:
            raise ScannerException(
                'Path {} does not exist'.format(metadata_path))

        data_path = '{}/tables/{}/{}_{}.bin'.format(
            self._db_path, self._table._descriptor.id, self._descriptor.id,
            item_id)
        try:
            data_file = RandomReadFile(self._storage, data_path)
        except UserWarning:
            raise ScannerException('Path {} does not exist'.format(path))

        # HACK: this should get eliminated once metadata format saves offsets instead of lengths
        last_row_edge_case = rows == [self._table._descriptor.end_rows[-1] - 1]
        if last_row_edge_case:
            size = metadata_file.size()
            metadata_file.seek(size - 8)
            (buf_len, ) = struct.unpack('=Q', metadata_file.read(8))
            data_file.seek(data_file.size() - buf_len)
            buf = data_file.read(buf_len)
            if len(buf) == 0:
                yield None
            elif fn is not None:
                yield fn(buf, self._db.protobufs)
            else:
                yield buf
            return

        sparse_load = len(rows) < LOAD_SPARSITY_THRESHOLD

        metadata_contents = metadata_file.read()
        if not sparse_load:
            data_contents = data_file.read()

        lens = []
        total_rows = 0
        i = 0
        while i < len(metadata_contents):
            (num_rows, ) = struct.unpack("=Q", metadata_contents[i:i + 8])
            total_rows += num_rows
            i += 8
            for fi in range(num_rows):
                (buf_len, ) = struct.unpack("=Q", metadata_contents[i:i + 8])
                lens.append(buf_len)
                i += 8

        start_pos = None
        pos = 0
        rows = rows if len(rows) > 0 else list(range(total_rows))
        for fi in range(total_rows):
            old_pos = pos
            pos += lens[fi]
            if start_pos is None:
                start_pos = old_pos

        rows_idx = 0
        i = start_pos
        for j, buf_len in enumerate(lens):
            if rows_idx < len(rows) and j == rows[rows_idx]:
                if sparse_load:
                    data_file.seek(i)
                    buf = data_file.read(buf_len)
                else:
                    buf = data_contents[i:i + buf_len]

                # len(buf) == 0 when element is null
                if len(buf) == 0:
                    yield None
                elif fn is not None:
                    yield fn(buf, self._db.protobufs)
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
        rows = list(range(total_rows)) if rows is None else rows
        prev = 0
        for item_id in range(num_items):
            start_row = prev
            end_row = table_descriptor.end_rows[item_id]
            item_rows = end_row - start_row
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
                    yield output
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

        self._load_meta()
        # If the column is a video, then dump the requested frames to disk as
        # PNGs and return the decoded PNGs
        if (self._descriptor.type == self._db.protobufs.Video
                and self._video_descriptor.codec_type ==
                self._db.protobufs.VideoDescriptor.H264):
            png_table_name = self._db._png_dump_prefix.format(
                self._table.name())
            if self._db.has_table(png_table_name):
                png_table = self._db.table(png_table_name)
                if rows is None and \
                   png_table.num_rows() == self._table.num_rows() and \
                   png_table._descriptor.timestamp > \
                   self._table._descriptor.timestamp:
                    return png_table.column('img').load(readers.image)
            pair = [(self._table.name(), png_table_name)]
            op_args = {}
            frame = self._db.sources.FrameColumn()
            op_args[frame] = self
            enc_input = frame
            if rows is not None:
                sampled_frame = self._db.streams.Gather(frame, rows=rows)
                enc_input = sampled_frame
            img = self._db.ops.ImageEncoder(frame=enc_input)
            output_op = self._db.sinks.Column(columns={'img': img})
            op_args[output_op] = png_table_name
            job = Job(op_args=op_args)
            [out_tbl] = self._db.run(
                output_op, [job], force=True, show_progress=False)
            return out_tbl.column('img').load(readers.image)
        elif self._descriptor.type == self._db.protobufs.Video:
            frame_type = self._video_descriptor.frame_type
            if frame_type == self._db.protobufs.U8:
                dtype = np.uint8
            elif frame_type == self._db.protobufs.F32:
                dtype = np.float32
            elif frame_type == self._db.protobufs.F64:
                dtype = np.float64
            parser_fn = readers.raw_frame_gen(
                self._video_descriptor.height, self._video_descriptor.width,
                self._video_descriptor.channels, dtype)
            return self._load(fn=parser_fn, rows=rows)
        else:
            return self._load(fn, rows=rows)

    def save_mp4(self, output_name, fps=None, scale=None):
        self._load_meta()
        if not (self._descriptor.type == self._db.protobufs.Video
                and self._video_descriptor.codec_type ==
                self._db.protobufs.VideoDescriptor.H264):
            raise ScannerException('Attempted to save a non-h264-compressed '
                                   'column as an mp4. Try compressing the '
                                   'column first by saving the output as '
                                   'an RGB24 frame')
        num_items = len(self._table._descriptor.end_rows)

        paths = [
            '{}/tables/{:d}/{:d}_{:d}.bin'.format(self._db._db_path,
                                                  self._table._descriptor.id,
                                                  self._descriptor.id, item_id)
            for item_id in range(num_items)
        ]
        temp_paths = []
        for _ in range(len(paths)):
            fd, p = tempfile.mkstemp()
            os.close(fd)
            temp_paths.append(p)
        # Copy all files locally before calling ffmpeg
        for in_path, temp_path in zip(paths, temp_paths):
            with open(temp_path, 'wb') as f:
                f.write(self._storage.read(in_path))

        files = '|'.join(temp_paths)

        vid_fps = (fps or (1.0 / (self._video_descriptor.time_base_num / float(
            self._video_descriptor.time_base_denom))))

        args = ''
        if scale:
            args += '-filter:v "scale={:d}x{:d}" '.format(scale[0], scale[1])

        cmd = (
            'ffmpeg -y '
            '-r {fps:f} '  # set the input fps
            '-i "concat:{input_files:s}" '  # concatenate the h264 files
            '-c:v libx264 '
            '-filter:v "setpts=N" '  # h264 does not have pts' in it
            '{extra_args:s}'
            '{output_name:s}.mp4'.format(
                input_files=files,
                fps=vid_fps,
                extra_args=args,
                output_name=output_name))
        rc = Popen(cmd, shell=True).wait()
        if rc != 0:
            raise ScannerException('ffmpeg failed during mp4 export!')
