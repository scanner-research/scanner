FROM scannerresearch/scanner-base:ubuntu16.04-cuda8.0-cv3.1.0
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"

ADD . /opt/scanner
WORKDIR /opt/scanner
RUN cd thirdparty && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    make -j
RUN mkdir build && cd build && \
    cmake -D BUILD_CAFFE_EVALUATORS=ON \
          -D BUILD_CAFFE_INPUT_EVALUATORS=ON \
          -D BUILD_UTIL_EVALUATORS=ON \
          -D BUILD_TESTS=ON \
          .. && \
    make -j
RUN mv .scanner.example.toml /root/.scanner.toml
