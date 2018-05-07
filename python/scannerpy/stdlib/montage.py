import cv2
import math
import numpy as np


def make_montage(n, frames, frame_width=64, frames_per_row=16):
    frame = next(frames)
    (frame_h, frame_w, _) = frame.shape
    target_w = frame_width
    target_h = int(target_w / float(frame_w) * frame_h)
    img_w = frames_per_row * target_w
    img_h = int(math.ceil(float(n) / frames_per_row)) * target_h
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    def place_image(i, fr):
        fr = cv2.resize(fr, (target_w, target_h))
        row = i / frames_per_row
        col = i % frames_per_row
        img[(row * target_h):((row+1) * target_h),
            (col * target_w):((col+1) * target_w),
            :] = fr

    place_image(0, frame)
    for i, frame in enumerate(frames):
        place_image(i + 1, frame)

    return img
