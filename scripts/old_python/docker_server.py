import socket
import logging
import knn
import json
import os
import random
from extract_frames_scanner import *
from movie_graphs import get_shot_type, get_shot_people, CLASSES, PERSON, classify_shots, get_shots
from decode import load_bboxes, db
from subprocess import check_output, STDOUT
import re
import sys

def main():
    dataset_name = 'anewhope'
    knn_job_name = 'frame_features'
    knn_patches_job_name = 'patch_features'
    dataset_meta = db.dataset_metadata(dataset_name).video_data
    vid_descriptors = db.video_descriptors(dataset_name)

    # KNN
    searcher = knn.FeatureSearch(dataset_name, knn_job_name)
    get_exemplar_features = knn.init_net()

    # Composition
    all_bboxes = list(load_bboxes(dataset_name, knn_patches_job_name).as_frame_list())
    stride = 8

    def draw_bboxes(vid, frame, name):
        img = cv2.imread('/bigdata/0_{:07d}.jpg'.format(frame))
        vid_bboxes = all_bboxes[vid][1]
        (f, bboxes) = vid_bboxes[frame/stride]
        print f
        for bbox in bboxes:
            if bbox.label == PERSON:
                print bbox
                cv2.rectangle(img,
                              (int(bbox.x1), int(bbox.y1)),
                              (int(bbox.x2), int(bbox.y2)),
                              (0, 0, 255),
                              thickness=3)
        cv2.imwrite('/bigdata/{}.jpg'.format(name), img)


    def get_diff_results(results, N):
        final = []
        for (vid, frame) in results[1:]:
            ignore = False
            for (good_vid, good_frame) in final:
                if vid == good_vid and abs(frame - good_frame) <= 300:
                    ignore = True
                    break
            if ignore: continue
            final.append((vid, frame))
            if len(final) == N: break
        return final

    def query_knn():
        exemplar = get_exemplar_features('/bigdata/query.png')
        results = searcher.search(exemplar)
        results = get_diff_results(results, 10)
        print(results)
        write_indices(results)
        extract_frames({
            'dataset': dataset_name,
            'out_dir': '/bigdata'
        })
        for (i, (vid, frame)) in enumerate(results):
            os.system('mv /bigdata/{}_{:07d}.jpg /bigdata/result{}.jpg' \
                      .format(vid, frame, i))

    def query_composition(shot_type, shot_people):
        results = []
        for (vid, vid_bboxes) in all_bboxes:
            desc = vid_descriptors[vid]
            vid_area = float(desc.width * desc.height)
            for (frame, frame_bboxes) in vid_bboxes:
                ppl = [p for p in frame_bboxes if p.label == PERSON]
                if len(ppl) == 0: continue
                areas = [(p.x2 - p.x1) * (p.y2 - p.y1) / vid_area for p in ppl]
                ty = get_shot_type(areas)
                ppl = get_shot_people(areas)
                if (shot_type == 'NA' or \
                    shot_type == ty) and \
                    (shot_people == 'NA' or \
                     shot_people == ppl):
                    results.append((vid, frame))
        print(shot_type, shot_people, len(results))
        random.seed(0xdeadbeef)
        random.shuffle(results)
        results = get_diff_results(results, 10)
        write_indices(results)
        extract_frames({
            'dataset': dataset_name,
            'out_dir': '/bigdata'
        })
        for (i, (vid, frame)) in enumerate(results):
            draw_bboxes(int(vid), frame, 'result{}'.format(i))

    def query_montage(movie):
        print 'Generating montage...'
        output = check_output('./scripts/generate_montage.sh {}'.format(movie),
                              shell=True,
                              stderr=STDOUT)
        times = [s.replace('elapsed', '') for s in re.findall(r'[^\s]+elapsed', output)]
        os.system('mv shot_montage_tmp_by_time.jpg /home/wcrichto/mp/movieproject-django/mpserver/mp/static/mp/shot_montage_time.jpg')
        os.system('mv shot_montage_tmp_by_color.jpg /home/wcrichto/mp/movieproject-django/mpserver/mp/static/mp/shot_montage_color.jpg')
        os.system('mv median_bar_montage_tmp.png /home/wcrichto/mp/movieproject-django/mpserver/mp/static/mp/median_montage.png')
        return times

    def query_shot_sequence(seq):
        dataset = 'anewhope'
        stride = 8
        cls, shots = classify_shots(dataset, stride)
        seq = seq.split(' ')
        N = len(seq)

        results = []
        for i in range(len(cls) - N + 1):
            if cls[i:i+N] == seq:
                #print cls[i:i+N], shots[i:i+N]
                results.append(i)

        random.seed(0xdeadbeef)
        random.shuffle(results)
        results = results[:10]

        final = []
        for i in results:
            final = final + [('0', shots[i+j]) for j in range(N)]

        write_indices(final)
        extract_frames({
            'dataset': dataset_name,
            'out_dir': '/bigdata'
        })

        for (a, i) in enumerate(results):
            for j in range(N):
                frame = shots[i+j]
                draw_bboxes(0, frame, 'result{}_{}'.format(a, j))

        return {
            'num_shots': len(results),
            'shot_len': N,
            'indices': [i for (_, i) in final]
        }


    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', port))
    s.listen(1)

    while True:
        conn, _ = s.accept()
        while True:
            data = conn.recv(1024)
            if not data: break
            query = json.loads(data)
            if query['key'] == 'knn':
                query_knn()
                conn.send(json.dumps([]))
            elif query['key'] == 'composition':
                params = query['value']
                query_composition(params['length'], params['type'])
                conn.send(json.dumps([]))
            elif query['key'] == 'montage':
                times = query_montage(query['value'])
                conn.send(json.dumps(times))
            elif query['key'] == 'shot_sequence':
                ret = query_shot_sequence(query['value'])
                conn.send(json.dumps(ret))
        conn.close()

if __name__ == "__main__":
    main()
