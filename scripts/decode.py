import json
import numpy as np
import struct
import sys

sys.path.append('build')
from metadata_pb2 import JobDescriptor

def load_output_buffers(job_name, column, fn, intvl=None):
    job = JobDescriptor()
    with open('db/{}_job_descriptor.bin'.format(job_name), 'r') as f:
        job.ParseFromString(f.read())

    videos = []
    for entry in job.columns:
        video = {'path': entry.name, 'buffers': []}
        (istart, iend) = intvl if intvl is not None else (0, sys.maxint)
        for intvl in entry.intervals:
            start = intvl.start
            end = intvl.end
            path = 'db/{}_job/{}_{}_{}-{}.bin'.format(job_name, entry.id, column, start, end)
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

def get_output_size(job_name):
    with open('db/{}_job_descriptor.bin'.format(job_name), 'r') as f:
        job = json.loads(f.read())

    return job['videos'][0]['intervals'][-1][1]

def load_histograms(job_name):
    def buf_to_histogram(buf):
        return np.split(np.frombuffer(buf, dtype=np.dtype(np.float32)), 3)
    return load_output_buffers(job_name, 'histogram', buf_to_histogram)

def load_faces(job_name):
    def buf_to_faces(buf):
        num_faces = len(buf) / 16
        faces = []
        for i in range(num_faces):
            faces.append(struct.unpack("iiii", buf[(16*i):(16*(i+1))]))
        return faces
    return load_output_buffers(job_name, 'faces', buf_to_faces)

def load_opticalflow(job_name):
    def buf_to_flow(buf):
        return np.frombuffer(buf, dtype=np.dtype(np.float32)).reshape((640, 320, 2))
    return load_output_buffers(job_name, 'opticalflow', buf_to_flow)

JOB = 'olivs'

def save_movie_info():
    np.save('{}_faces.npy'.format(JOB), load_faces(JOB)[0]['buffers'])
    np.save('{}_histograms.npy'.format(JOB), load_histograms(JOB)[0]['buffers'])
    np.save('{}_opticalflow.npy'.format(JOB), load_opticalflow(JOB)[0]['buffers'])

# After running this, run:
# ffmpeg -safe 0 -f concat -i <(for f in ./*.mkv; do echo "file '$PWD/$f'"; done) -c copy output.mkv
def save_debug_video():
    bufs = load_output_buffers(JOB, 'video', lambda buf: buf)[0]['buffers']
    i = 0
    for buf in bufs:
        if len(buf) == 0: continue
        with open('out__{:06d}.mkv'.format(i), 'wb') as f:
            f.write(buf)
        i += 1

def main():
    save_debug_video()

if __name__ == "__main__":
    main()
