FROM scannerresearch/scanner-base:ubuntu16.04-cuda8.0-cv3.2.0
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"
ARG cores=1

ADD . /opt/scanner
WORKDIR /opt/scanner
RUN cd thirdparty && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release .. && \
    make -j ${cores}
RUN mkdir build && cd build && \
    cmake -D BUILD_CAFFE_OPS=ON \
          -D BUILD_CAFFE_INPUT_OPS=ON \
          -D BUILD_UTIL_OPS=ON \
          -D BUILD_TESTS=ON \
          .. && \
    make -j ${cores}
RUN mkdir build_cpu && cd build_cpu && \
    cmake -D BUILD_CAFFE_OPS=ON \
          -D BUILD_CAFFE_INPUT_OPS=ON \
          -D BUILD_UTIL_OPS=ON \
          -D BUILD_TESTS=ON \
          -D BUILD_CUDA=OFF \
          .. && \
    make -j ${cores}
RUN mv .scanner.example.toml /root/.scanner.toml
