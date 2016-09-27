from __future__ import print_function
import json
import numpy as np
import struct
import cv2
import scipy.io as sio

import caffe

CRAIG_VID = '20140914_150520_697'
KENNYWOOD_VID = '20140913_184839_517'
VID = KENNYWOOD_VID
VIDEO = '/disk0/kcam/' + VID

def extract_features(net, frame):
    frame = np.array(frame, dtype=np.float32) # u8 -> f32
    frame -= np.array((119.29959869, 110.54627228, 101.8384321)) # sub mean
    frame = frame.transpose((2, 0, 1))

    net.blobs['data'].reshape(1, *frame.shape)
    net.blobs['data'].data[...] = frame
    net.forward()
    return (net.blobs['score_cls'].data[0], net.blobs['score_reg'].data[0])

def nms(bboxes, overlap=0.3):
    x1 = bboxes[0,:]
    y1 = bboxes[1,:]
    x2 = bboxes[2,:]
    y2 = bboxes[3,:]
    s = bboxes[4,:]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)

    pick = (s * 0).astype(int)
    counter = 0

    while len(I) > 0:
        last = len(I)
        i = I[last-1]
        pick[counter] = i
        counter += 1

        xx1 = np.maximum(x1[i], x1[I[0:last-2]])
        yy1 = np.maximum(y1[i], y1[I[0:last-2]])
        xx2 = np.maximum(x2[i], x2[I[0:last-2]])
        yy2 = np.maximum(y2[i], y2[I[0:last-2]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        o = np.divide(np.multiply(w, h), area[I[0:last-2]])

        nz = np.nonzero(o > overlap)

        I = np.delete(I, np.append(nz, [last - 1]))
    return bboxes[:,pick[0:counter - 1]]

def to_bboxes(shape, templates, f, threshold):
    score_class = f[0].transpose((0, 1, 2))
    score_reg = f[1].transpose((0, 1, 2))
    width = shape[2]
    height = shape[1]

    [fc, fy, fx] = np.nonzero(score_class > threshold)

    cy = fy * 8 - 2
    cx = fx * 8 - 2
    ch = templates[fc, 3] - templates[fc, 1] + 1
    cw = templates[fc, 2] - templates[fc, 0] + 1

    x1 = cx - cw / 2
    y1 = cy - ch / 2
    x2 = cx + cw / 2
    y2 = cy + ch / 2

    num_templates = 25
    tx = score_reg[0:num_templates,:,:]
    ty = score_reg[num_templates:num_templates * 2,:,:]
    tw = score_reg[num_templates * 2:num_templates * 3,:,:]
    th = score_reg[num_templates * 3:num_templates * 4,:,:]

    dcx = np.multiply(cw, tx[fc, fy, fx])
    dcy = np.multiply(ch, ty[fc, fy, fx])
    rcx = np.add(cx, dcx)
    rcy = np.add(cy, dcy)
    rcw = np.multiply(cw, np.exp(tw[fc, fy, fx]))
    rch = np.multiply(ch, np.exp(th[fc, fy, fx]))

    scores = score_class[fc, fy, fx]
    bboxes = np.array([rcx - rcw/2,
                       rcy - rch/2,
                       rcx + rcw/2,
                       rcy + rch/2,
                       scores])

    return bboxes

import time

def main():
    inp = cv2.VideoCapture('{}.mp4'.format(VIDEO))
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('{}_face.avi'.format(VID),
                          fourcc,
                          inp.get(cv2.cv.CV_CAP_PROP_FPS),
                          (1280,720))

    templates = sio.loadmat('facenet_templates.mat')
    templates = templates['templates']

    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net('facenet_deploy.prototxt',
                    'facenet_deploy.caffemodel',
                    caffe.TEST)

    i = 0

    while (inp.isOpened() and i <= 3000):
        _, frame = inp.read()
        if i % 100 == 0:
            print('Frame', i)

        millis = int(round(time.time() * 1000))
        features = extract_features(net, frame)
        end_millis = int(round(time.time() * 1000))
        print('ms', end_millis - millis)
        bboxes = to_bboxes(frame.shape, templates, features, 2.5)
        best_bboxes = nms(bboxes)
        if (len(best_bboxes[0,:]) > 0):
            print('Bboxes detected on frame ' + str(i) + ':',
                  len(best_bboxes[0,:]))

        x1 = np.maximum(best_bboxes[0,:], 0).astype(int)
        y1 = np.maximum(best_bboxes[1,:], 0).astype(int)
        x2 = np.minimum(best_bboxes[2,:], frame.shape[1] - 1).astype(int)
        y2 = np.minimum(best_bboxes[3,:], frame.shape[0] - 1).astype(int)
        for n in range(len(x1)):
            # face = frame[y1[n]:y2[n],x1[n]:x2[n]]
            cv2.rectangle(frame, (x1[n], y1[n]), (x2[n], y2[n]), (255, 0, 0), 2)
            # frame[y1[n]:y2[n],x1[n]:x2[n]] = (
            #     cv2.GaussianBlur(face, (0, 0), 5))
        out.write(frame)
        i += 1

    inp.release()
    out.release()

if __name__ == "__main__":
    main()
