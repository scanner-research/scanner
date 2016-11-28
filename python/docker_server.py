import socket
import logging
import knn
import json
import os
import random
from extract_frames_scanner import *
from decode import load_bboxes, db

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle',
           'airplane', 'bus','train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
           'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase',
           'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table','toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier','toothbrush')
PERSON = CLASSES.index('person')

ZOOMS = {
    "CU": 0.6,
    "MS": 0.05,
    "LS": 0
}
ZOOMS_SORT = sorted(ZOOMS, key=ZOOMS.get, reverse=True)


def get_zoom(areas):
    area = max(areas)
    for zoom in ZOOMS_SORT:
        if area > ZOOMS[zoom]:
            return zoom


def get_type(areas):
    if len(areas) == 1:
        return '1'
    elif len(areas) == 2:
        if abs(areas[0] - areas[1]) < 0.4:
            return '2-eq'
        else:
            return '2-neq'
    else:
        return 'n'


def main():
    dataset_name = 'meangirls'
    knn_job_name = 'frame_features'
    knn_patches_job_name = 'patch_features'
    dataset_meta = db.dataset_metadata(dataset_name).video_data
    vid_descriptors = db.video_descriptors(dataset_name)

    # KNN
    searcher = knn.FeatureSearch(dataset_name, knn_job_name)
    get_exemplar_features = knn.init_net()

    # Composition
    all_bboxes = list(load_bboxes(dataset_name, knn_patches_job_name).as_frame_list())

    def query_knn():
        exemplar = get_exemplar_features('/bigdata/query.png')
        results = searcher.search(exemplar)
        final = results[0:1]
        for (vid, frame) in results[1:]:
            ignore = False
            for (good_vid, good_frame) in final:
                if vid == good_vid and abs(frame - good_frame) <= 200:
                    ignore = True
                    break
            if ignore: continue
            final.append((vid, frame))
            if len(final) == 5: break
        results = final

        print results
        write_indices(results)
        extract_frames({
            'dataset': dataset_name,
            'out_dir': '/bigdata'
        })
        for (i, (vid, frame)) in enumerate(results):
            os.system('mv /bigdata/{}_{:07d}.jpg /bigdata/result{}.jpg' \
                      .format(vid, frame, i))

    def query_composition(shot_length, shot_type):
        results = []
        for (vid, vid_bboxes) in all_bboxes:
            desc = vid_descriptors[vid]
            vid_area = float(desc.width * desc.height)
            for (frame, frame_bboxes) in vid_bboxes:
                ppl = [p for p in frame_bboxes if p.label == PERSON]
                if len(ppl) == 0: continue
                areas = [(p.x2 - p.x1) * (p.y2 - p.y1) / vid_area for p in ppl]
                if (shot_length == 'NA' or \
                    shot_length == get_zoom(areas)) and \
                    (shot_type == 'NA' or \
                     shot_type == get_type(areas)):
                    results.append((vid, frame))
        print shot_length, shot_type, len(results)
        random.shuffle(results)
        results = results[:10]
        write_indices(results)
        extract_frames({
            'dataset': dataset_name,
            'out_dir': '/bigdata'
        })
        for (i, (vid, frame)) in enumerate(results):
            os.system('mv /bigdata/{}_{:07d}.jpg /bigdata/result{}.jpg' \
                      .format(vid, frame, i))


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 7000))
    s.listen(1)

    while True:
        conn, _ = s.accept()
        while True:
            data = conn.recv(1024)
            if not data: break
            query = json.loads(data)
            if query['key'] == 'knn':
                query_knn()
                conn.send('ack')
            else:
                params = query['value']
                query_composition(params['length'], params['type'])
                conn.send('ack')
        conn.close()

if __name__ == "__main__":
    main()
