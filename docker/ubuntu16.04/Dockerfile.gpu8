# Scanner base GPU image for Ubuntu 16.04 CUDA 8.0

ARG base_tag
FROM ${base_tag}
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"
ARG cores=1

ADD thirdparty/resources/cuda/libnvcuvid.so.367.48 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
RUN ln -s /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so \
       /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so.1
ENV CUDA_LIB_PATH /usr/local/cuda/lib64/stubs

RUN bash ./deps.sh --install-all --prefix /usr/local --use-gpu && \
    rm -rf /opt/scanner-base

ENV LD_LIBRARY_PATH /usr/local/intel/mkl/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH /usr/local/python:${PYTHONPATH}
ENV PYTHONPATH /usr/local/lib/python3.5/site-packages:${PYTHONPATH}
ENV PYTHONPATH /usr/local/lib/python3.5/dist-packages:${PYTHONPATH}

WORKDIR /
