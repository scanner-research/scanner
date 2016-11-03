from enum import Enum
from collections import defaultdict
import struct
import sys
import toml
import os
import logging
import sys

is_py3 = sys.version_info.major == 3
maxint = sys.maxsize if is_py3 else sys.maxint

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


class ScannerConfig(object):
    """ TODO(wcrichto): document me """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self.default_config_path()
        config = self.load_config(config_path)
        try:
            self.db_path = config['db_path']
            self.scanner_path = config['scanner_path']
        except KeyError as key:
            logging.critical('Scanner config missing key: {}'.format(key))
            exit()

    def default_config_path(self):
        return '{}/.scanner.toml'.format(os.path.expanduser('~'))

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return toml.loads(f.read())
        except IOError:
            logging.critical('Error: you need to setup your Scanner config. Run `python scripts/setup.py`.')
            exit()


class Sampling(Enum):
    All = 0
    Strided = 1
    Gather = 2
    SequenceGather = 3


class JobResult(object):
    """ TODO(apoms): document me """

    def __init__(self, scanner, dataset_name, job_name, column, load_fn):
        self._scanner = scanner
        self._dataset_name = dataset_name
        self._job_name = job_name
        self._column = column
        self._load_fn = load_fn

        self._db_path = self._scanner.config.db_path

        self._dataset = self._scanner._meta.DatasetDescriptor()
        with open('{}/{}_dataset_descriptor.bin'.format(self._db_path,
                                                        dataset_name),
                  'rb') as f:
            self._dataset.ParseFromString(f.read())

        self._job = self._scanner._meta.JobDescriptor()
        with open('{}/{}_job_descriptor.bin'.format(self._db_path, job_name),
                  'rb') as f:
            self._job.ParseFromString(f.read())

    def _load_output_file(self, video, video_name, work_item_index, rows,
                          istart, iend):
        path = '{}/{}_job/{}_{}_{}.bin'.format(
            self._db_path, self._job_name, video_name, self._column,
            work_item_index)
        try:
            with open(path, 'rb') as f:
                lens = []
                start_pos = maxint
                pos = 0
                for fi in rows:
                    byts = f.read(8)
                    (buf_len,) = struct.unpack("l", byts)
                    old_pos = pos
                    pos += buf_len
                    if (fi >= istart and fi <= iend):
                        if start_pos == maxint:
                            start_pos = old_pos
                        lens.append(buf_len)

                bufs = []
                f.seek(len(rows) * 8 + start_pos)
                for buf_len in lens:
                    buf = f.read(buf_len)
                    item = self._load_fn(buf, video)
                    bufs.append(item)

                return bufs
        except IOError as err:
            logging.critical(err)
            exit()

    def _load_video_descriptor(self, video_name):
        video = self._scanner._meta.VideoDescriptor()
        with open('{}/{}_dataset/{}_metadata.bin'
                  .format(self._db_path, self._dataset_name, video_name),
                  'rb') as f:
            video.ParseFromString(f.read())
        return video

    def _load_all_sampling(self, interval=None):
        item_size = self._job.work_item_size
        work_item_index = 0
        for vi, video_name in enumerate(self._dataset.video_names):
            video = self._load_video_descriptor(video_name)

            intervals = [i for i in range(video.frames - 1, item_size)]
            intervals.append(video.frames)
            intervals = zip(intervals[:-1], intervals[1:])
            assert(intervals is not None)

            result = {'video': video_name,
                      'frames': [],
                      'buffers': []}
            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for i, ivl in enumerate(intervals):
                start = ivl[0]
                end = ivl[1]
                if start > iend or end < istart: continue
                result['buffers'] += self._load_output_file(video,
                                                            video_name,
                                                            work_item_index,
                                                            range(start, end),
                                                            istart,
                                                            iend)
                result['frames'] += range(start, end)
                work_item_index += 1
            yield result

    def _load_stride_sampling(self, interval=None):
        item_size = self._job.work_item_size
        stride = self._job.stride
        work_item_index = 0
        for vi, video_name in enumerate(self._dataset.video_names):
            video = self._load_video_descriptor(video_name)

            intervals = [i for i in range(0, video.frames - 1, item_size * stride)]
            intervals.append(video.frames)
            intervals = zip(intervals[:-1], intervals[1:])
            assert(intervals is not None)

            result = {'video': video_name,
                      'frames': [],
                      'buffers': []}
            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for i, ivl in enumerate(intervals):
                start = ivl[0]
                end = ivl[1]
                if start > iend or end < istart: continue
                rows = (end - start) / stride
                rows = range(start, end, stride)
                result['buffers'] += self._load_output_file(video,
                                                            video_name,
                                                            work_item_index,
                                                            rows,
                                                            istart,
                                                            iend)
                result['frames'] += range(start, end, stride)
                work_item_index += 1
            yield result

    def _load_gather_sampling(self, interval=None):
        item_size = self._job.work_item_size
        work_item_index = 0
        for samples in self._job.gather_points:
            video_index = samples.video_index
            video_name = self._dataset.video_names[video_index]
            video = self._load_video_descriptor(video_name)

            work_items = chunks(samples.frames, item_size)
            assert(work_items is not None)

            result = {'video': video_name,
                      'frames': [],
                      'buffers': []}
            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for i, item in enumerate(work_items):
                start = item[0]
                end = item[-1]
                if start > iend or end < istart: continue
                rows = item
                result['buffers'] += self._load_output_file(video,
                                                            video_name,
                                                            work_item_index,
                                                            rows,
                                                            istart,
                                                            iend)
                result['frames'] += item
                work_item_index += 1
            yield result

    def _load_sequence_gather_sampling(self, interval=None):
        item_size = self._job.work_item_size
        work_item_index = 0
        for samples in self._job.gather_sequences:
            video_index = samples.video_index
            video_name = self._dataset.video_names[video_index]
            video = self._load_video_descriptor(video_name)

            sequences = samples.intervals

            result = {'video': video_name,
                      'sequences': [(s.start, s.end) for s in sequences],
                      'frames': [],
                      'buffers': []}

            (istart, iend) = interval if interval is not None else (0,
                                                                    maxint)
            for intvl in sequences:
                intervals = [i for i in range(intvl.start, intvl.end - 1,
                                              item_size)]
                intervals.append(intvl.end)
                intervals = zip(intervals[:-1], intervals[1:])
                assert(intervals is not None)

                print(intervals)
                for i, intvl in enumerate(intervals):
                    start = intvl[0]
                    end = intvl[1]
                    if start > iend or end < istart: continue
                    rows = range(start, end)
                    result['buffers'] += self._load_output_file(video,
                                                                video_name,
                                                                work_item_index,
                                                                rows,
                                                                istart,
                                                                iend)
                    result['frames'] += range(start, end)
                    work_item_index += 1
            yield result

    def get_sampling_type(self):
        JD = self._scanner._meta.JobDescriptor
        js = self._job.sampling
        s = None
        if js == JD.All:
            s = Sampling.All
        elif js == JD.Strided:
            s = Sampling.Strided
        elif js == JD.Gather:
            s = Sampling.Gather
        elif js == JD.SequenceGather:
            s = Sampling.SequenceGather
        return s

    def as_outputs(self, interval=None):
        sampling = self.get_sampling_type()
        if sampling is Sampling.All:
            return self._load_all_sampling(interval)
        elif sampling is Sampling.Strided:
            return self._load_stride_sampling(interval)
        elif sampling is Sampling.Gather:
            return self._load_gather_sampling(interval)
        elif sampling is Sampling.SequenceGather:
            return self._load_sequence_gather_sampling(interval)

    def as_frame_list(self, interval=None):
        for d in self.as_outputs(interval):
            yield (d['video'], zip(d['frames'], d['buffers']))

    def as_sequences(self, interval=None):
        """ TODO(apoms): implement """
        if self.get_sampling_type() != Sampling.SequenceGather:
            logging.error("")
            return
        pass

class Scanner(object):
    """ TODO(wcrichto): document me """

    def __init__(self, config_path=None):
        self.config = ScannerConfig(config_path)
        sys.path.append('{}/build'.format(self.config.scanner_path))
        from scannerpy import metadata_pb2
        self._meta = metadata_pb2

    def get_job_result(self, dataset_name, job_name, column, fn):
        return JobResult(self, dataset_name, job_name, column, fn)

    def write_output_buffers(dataset_name, job_name, ident, column, fn, video_data):
        job = self._meta.JobDescriptor()
        job.id = ident

        col = job.columns.add()
        col.id = 0
        col.name = column

        for i, data in enumerate(video_data):
            start = 0
            end = len(data)
            video = job.videos.add()
            video.index = i
            interval = video.intervals.add()
            interval.start = start
            interval.end = end

            path = '{}/{}_job/{}_{}_{}-{}.bin'.format(
                self.config.db_path, job_name, str(i), column, start, end)

            with open(path, 'wb') as f:
                all_bytes = ""
                for d in data:
                    byts = fn(d)
                    all_bytes += byts
                    f.write(struct.pack("=Q", len(byts)))
                f.write(all_bytes)

        with open('{}/{}_job_descriptor.bin'.format(self.config.db_path, job_name), 'wb') as f:
            f.write(job.SerializeToString())

    def get_output_size(self, job_name):
        with open('{}/{}_job_descriptor.bin'.format(self.config.db_path, job_name), 'r') as f:
            job = json.loads(f.read())

        return job['videos'][0]['intervals'][-1][1]

    def loader(self, column):
        def decorator(f):
            def loader(dataset_name, job_name):
                return self.get_job_result(dataset_name, job_name, column, f)
            return loader
        return decorator

    def load_db_metadata(self):
        path = '{}/db_metadata.bin'.format(self.config.db_path)
        meta = self._meta.DatabaseDescriptor()
        with open(path, 'rb') as f:
            meta.ParseFromString(f.read())
        return meta

    def write_db_metadata(self, meta):
        path = '{}/db_metadata.bin'.format(self.config.db_path)
        with open(path, 'wb') as f:
            f.write(meta.SerializeToString())
