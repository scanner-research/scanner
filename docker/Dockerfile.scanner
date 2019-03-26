ARG tag=gpu
FROM scannerresearch/scanner-base:ubuntu16.04-${tag}
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"
ARG cores=1
ARG gpu=ON
ARG deps_opt=''

ADD . /opt/scanner
WORKDIR /opt/scanner
ENV Caffe_DIR /usr/local
ENV LD_LIBRARY_PATH \
       "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs"
ENV PKG_CONFIG_PATH "/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
RUN cd /opt/scanner && \
    (if [ "${gpu}" = "ON" ]; then \
     bash deps.sh -g --install-none --prefix /usr/local ${deps_opt}; \
     else \
     bash deps.sh -ng --install-none --prefix /usr/local ${deps_opt}; \
     fi) && \
    mkdir build && cd build && \
    cmake -D BUILD_TESTS=ON \
          -D BUILD_CUDA=${gpu} \
          -D CMAKE_BUILD_TYPE=RelWithDebinfo \
          .. && \
    cd .. && \
    (yes | pip3 uninstall grpcio protobuf) && \
    ./build.sh && \
    ldconfig

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
