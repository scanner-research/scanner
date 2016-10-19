from __future__ import print_function
import json
import numpy as np
import struct
import sys
from pprint import pprint
import metadata_pb2
import evaluators.types_pb2

DB_PATH = '/data/apoms/scanner_db'

def read_string(f):
    s = ""
    while True:
        byts = f.read(1)
        (c,) = struct.unpack("c", byts)
        if c == '\0':
            break
        s += c
    return s


def load_db_metadata():
    path = DB_PATH + '/db_metadata.bin'
    meta = metadata_pb2.DatabaseDescriptor()
    with open(path, 'rb') as f:
        meta.ParseFromString(f.read())
    return meta


def write_db_metadata(meta):
    path = DB_PATH + '/db_metadata.bin'
    with open(path, 'wb') as f:
        f.write(meta.SerializeToString())


def load_output_buffers(dataset_name, job_name, column, fn, intvl=None):
    dataset = metadata_pb2.DatasetDescriptor()
    with open(DB_PATH + '/{}_dataset_descriptor.bin'.format(dataset_name), 'rb') as f:
        dataset.ParseFromString(f.read())

    job = metadata_pb2.JobDescriptor()
    with open(DB_PATH + '/{}_job_descriptor.bin'.format(job_name), 'rb') as f:
        job.ParseFromString(f.read())

    videos = []
    for v_index, json_video in enumerate(dataset.video_names):
        video = {'path': str(v_index), 'buffers': []}
        (istart, iend) = intvl if intvl is not None else (0, sys.maxint)
        intervals = None
        for v in job.videos:
            if v.index == v_index:
                intervals = v.intervals
        assert(intervals is not None)
        for interval in intervals:
            start = interval.start
            end = interval.end
            path = DB_PATH + '/{}_job/{}_{}_{}-{}.bin'.format(
                job_name, str(v_index), column, start, end)
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
                        item = fn(buf)
                        bufs.append(item)

                    video['buffers'] += bufs
            except IOError as e:
                print("{}".format(e))
        videos.append(video)

    return videos


def write_output_buffers(dataset_name, job_name, ident, column, fn, video_data):
    job = metadata_pb2.JobDescriptor()
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

        path = DB_PATH + '/{}_job/{}_{}_{}-{}.bin'.format(
            job_name, str(i), column, start, end)

        with open(path, 'wb') as f:
            all_bytes = ""
            for d in data:
                byts = fn(d)
                all_bytes += byts
                f.write(struct.pack("=Q", len(byts)))
            f.write(all_bytes)

    with open(DB_PATH + '/{}_job_descriptor.bin'.format(job_name), 'wb') as f:
        f.write(job.SerializeToString())


def get_output_size(job_name):
    with open(DB_PATH + '/{}_job_descriptor.bin'.format(job_name), 'r') as f:
        job = json.loads(f.read())

    return job['videos'][0]['intervals'][-1][1]
