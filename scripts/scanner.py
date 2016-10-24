import struct
import sys
import toml
import os
import logging

class ScannerConfig:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self.default_config_path()
        config = self.load_config(config_path)
        try:
            self.db_path = config['db_path']
            self.scanner_path = config['scanner_path']
        except KeyError as key:
            logging.critical('Scanner config missing key: {}'.format(key))

    def default_config_path(self):
        return '{}/.scanner.toml'.format(os.path.expanduser('~'))

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return toml.loads(f.read())
        except IOError:
            logging.critical('Error: you need to setup your Scanner config. Run `python scripts/setup.py`.')
            exit()

class Scanner:
    def __init__(self, config_path=None):
        self.config = ScannerConfig(config_path)
        sys.path.append('{}/build'.format(self.config.scanner_path))
        import metadata_pb2
        self._meta = metadata_pb2


    def load_output_buffers(self, dataset_name, job_name, column, fn, intvl=None):
        db_path = self.config.db_path

        dataset = self._meta.DatasetDescriptor()
        with open('{}/{}_dataset_descriptor.bin'.format(db_path, dataset_name), 'rb') as f:
            dataset.ParseFromString(f.read())

        job = self._meta.JobDescriptor()
        with open('{}/{}_job_descriptor.bin'.format(db_path, job_name), 'rb') as f:
            job.ParseFromString(f.read())

        for v_index, json_video in enumerate(dataset.video_names):
            video = self._meta.VideoDescriptor()
            with open('{}/{}_dataset/{}_metadata.bin'.format(db_path, dataset_name, v_index), 'rb') as f:
                video.ParseFromString(f.read())

            result = {'path': str(v_index), 'buffers': []}
            intervals = None
            for v in job.videos:
                if v.index == v_index:
                    intervals = v.intervals
            assert(intervals is not None)

            (istart, iend) = intvl if intvl is not None else (0, sys.maxint)
            for ivl in intervals:
                start = ivl.start
                end = ivl.end
                path = '{}/{}_job/{}_{}_{}-{}.bin'.format(
                    db_path, job_name, v_index, column, start, end)
                if start > iend or end < istart: continue
                try:
                    with open(path, 'rb') as f:
                        lens = []
                        start_pos = sys.maxint
                        pos = 0
                        for i in range(end-start):
                            idx = i + start
                            byts = f.read(8)
                            (buf_len,) = struct.unpack("l", byts)
                            old_pos = pos
                            pos += buf_len
                            if (idx >= istart and idx <= iend):
                                if start_pos == sys.maxint:
                                    start_pos = old_pos
                                lens.append(buf_len)

                        bufs = []
                        f.seek((end-start) * 8 + start_pos)
                        for buf_len in lens:
                            buf = f.read(buf_len)
                            item = fn(buf, video)
                            bufs.append(item)

                        result['buffers'] += bufs
                except IOError as err:
                    logging.warning(err)
            yield result


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
                return self.load_output_buffers(dataset_name, job_name, column, f)
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
