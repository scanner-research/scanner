FROM scannerresearch/scanner-base:ubuntu16.04-cuda8.0-cv3.1.0
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"

ADD . /opt/scanner
WORKDIR /opt/scanner
RUN cd thirdparty && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    make
RUN mkdir build && cd build && \
    cmake -D PIPELINE_FILE=../scanner/pipelines/knn_pipeline.cpp  \
          -D BUILD_CAFFE_EVALUATORS=ON \
          -D BUILD_CAFFE_INPUT_EVALUATORS=ON \
          -D BUILD_UTIL_EVALUATORS=ON \
          .. && \
    make
RUN ./features/squeezenet.sh
RUN echo '\n\
scanner_path = "/opt/scanner" \n\
\n\
[storage] \n\
    type = "posix" \n\
    db_path = "/opt/scanner_db" \n\
\n\
[job]\n\
    work_item_size = 96' > /root/.scanner.toml
