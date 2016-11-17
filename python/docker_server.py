import socket
import logging
import knn
import json
import os
from extract_frames_scanner import *

def main():
    dataset_name = 'meangirls'
    job_name = 'mean'
    searcher = knn.FeatureSearch(dataset_name, job_name)
    get_exemplar_features = knn.init_net()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 7000))
    s.listen(1)

    while True:
        conn, _ = s.accept()
        while True:
            data = conn.recv(1024)
            if not data: break
            exemplar = get_exemplar_features('/bigdata/query.png')
            results = searcher.search(exemplar)
            write_indices(results)
            extract_frames(dataset_name)
            for (i, (vid, frame)) in enumerate(results):
                os.system('mv /bigdata/{}_{}.jpg /bigdata/result{}.jpg' \
                          .format(vid, frame, i))
            conn.send('ack')
        conn.close()

if __name__ == "__main__":
    main()
