from mpi4py import MPI
from decode import load_output_buffers, get_output_size
import numpy as np
import cv2

IS_COLOR = True
JOB = 'mean'
VIDEO = '/homes/wcrichto/lightscan/meanGirls'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
chunk_size = get_output_size(JOB) / size
start = rank * chunk_size
end = start + chunk_size

def load_images(job_name):
    def buf_to_image(buf):
        return np.frombuffer(buf, dtype=np.dtype(np.uint8)).reshape((480, 640, 3))
    return load_output_buffers(job_name, 'face', buf_to_image, intvl=(start, end))

images = load_images(JOB)[0]['buffers']
inp = cv2.VideoCapture('{}.mp4'.format(VIDEO))
out = cv2.VideoWriter(
    '{}_{:06d}-{:06}.mkv'.format(VIDEO, start, end),
    cv2.VideoWriter_fourcc(*'X264'),
    inp.get(cv2.CAP_PROP_FPS),
    (640, 480),
    IS_COLOR)

print(start, end, len(images))

for img in images:
    if IS_COLOR: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)

inp.release()
out.release()
