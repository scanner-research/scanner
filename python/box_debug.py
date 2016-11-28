from decode import load_bboxes, db
from fast_rcnn.nms_wrapper import nms
import numpy as np
import cv2

@db.loader('rois')
def load_rois(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    return np.split(buf, len(buf) / 5)

@db.loader('cls_prob')
def load_prob(buf, metadata):
    buf = np.frombuffer(buf, dtype=np.dtype(np.float32))
    return np.split(buf, len(buf) / 21)

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle',
           'airplane', 'bus','train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier','toothbrush')

def main():
    inp = cv2.VideoCapture('/bigdata/wcrichto/videos/bourne_short.mp4')
    out = cv2.VideoWriter('bourne_new.mkv', cv2.VideoWriter_fourcc(*'H264'), 24.0, (640, 360))

    if False:
        for ((_, vid_rois), (_, vid_probs)) in zip(
                load_rois('bourne', 'bour').as_frame_list(),
                load_prob('bourne', 'bour').as_frame_list()):
            for ((_, frame_rois), (_, frame_probs)) in zip(vid_rois, vid_probs):
                _, frame = inp.read()
                scores = np.array(frame_probs)
                boxes = np.array(frame_rois)[:,1:]
                boxes = np.tile(boxes, (1, scores.shape[1]))
                for cls_ind, cls in enumerate(CLASSES[1:]):
                    cls_ind += 1
                    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                    cls_scores = scores[:, cls_ind]
                    dets = np.hstack((cls_boxes,
                                      cls_scores[:, np.newaxis])).astype(np.float32)
                    keep = nms(dets, 0.3)
                    dets = dets[keep, :]

                    inds = np.where(dets[:,-1] >= 0.8)[0]
                    for i in inds:
                        roi = dets[i, :4]
                        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]),
                                      (0, 0, 255))
                out.write(frame)
    else:
        for (_, vid_boxes) in load_bboxes('bourne', 'patch_features').as_frame_list():
            print len(vid_boxes)
            for (_, frame_boxes) in vid_boxes:
                _, frame = inp.read()
                for box in frame_boxes:
                    box = map(int, [box.x1, box.y1, box.x2, box.y2])
                    cv2.rectangle(frame, tuple(box[0:2]), tuple(box[2:4]),
                                  (0, 0, 255))
                out.write(frame)



if __name__ == "__main__":
    main()
