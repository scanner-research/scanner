FROM scannerresearch/scanner-base:ubuntu16.04-cuda8.0-cv3.2.0
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"
ARG cores=1
ARG gpu=ON

ADD . /opt/scanner
WORKDIR /opt/scanner
RUN cd thirdparty && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    make -j ${cores}
RUN mkdir build && cd build && \
    cmake -D BUILD_CAFFE_OPS=ON \
          -D BUILD_IMGPROC_OPS=ON \
          -D BUILD_TESTS=ON \
          -D BUILD_CUDA=${gpu} \
          .. && \
    make -j ${cores}
ENV PYTHONPATH /opt/scanner/python:$PYTHONPATH
