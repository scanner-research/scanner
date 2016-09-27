import json
import numpy as np
import struct
import cv2

import caffe

DB = '/disk0/apoms/scanner_db'
VIDEO = '/disk0/kcam/'
JOB = 'foo'

def extract_features(net, frame):
    frame = np.array(frame, dtype=np.float32) # u8 -> f32
    frame -= np.array((119.29959869, 110.54627228, 101.8384321)) # sub mean
    frame = frame.transpose((2, 1, 0)) 

    net.blobs['data'].reshape(1, *frame.shape)
    net.blobs['data'].data[...] = frame
    net.forward()
    return net.blobs['score_final'].data[0]

def nms(bboxes, overlap=0.3):
    pass

def to_bboxes(shape, f):
    print(shape)
    return []

def main():
    inp = cv2.VideoCapture('{}.mp4'.format(VIDEO))
    fourcc = cv2.cv.CV_FOURCC(*'avc1')
    out = cv2.VideoWriter('{}_face.mov'.format(VIDEO),
                          fourcc,
                          inp.get(cv2.cv.CV_CAP_PROP_FPS),
                          (1280,720))

    net = caffe.Net('facenet_deploy.prototxt',
                    'facenet_deploy.caffemodel',
                    caffe.TEST)

    i = 0

    print(len(faces))

    while (inp.isOpened() and i <= 10000):
        _, frame = inp.read()
        print(i)
        features = extract_features(net, frame)
        bboxes = to_bboxes(frame.shape, features)
        for [x, y, w, h] in bboxes:
            x1 = x - (w / 2) * 1.2
            y1 = y - (h / 2) * 1.2
            x2 = x + (w / 2) * 1.2
            y2 = y + (h / 2) * 1.2
            frame[y1:y2,x1:x2] = (
                cv2.GaussianBlur(frame[y1:y2,x1:x2], (0, 0), 5))
        out.write(frame)
        i += 1

    inp.release()
    out.release()

if __name__ == "__main__":
    main()
