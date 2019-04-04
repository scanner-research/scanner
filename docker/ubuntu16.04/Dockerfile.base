# Scanner base image for Ubuntu 16.04

ARG base_tag
FROM ${base_tag}
MAINTAINER Will Crichton "wcrichto@cs.stanford.edu"
ARG cores=1
ARG cpu_only=OFF

# Apt-installable dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt-get update && \
    apt-get install -y \
      build-essential \
      git libgtk2.0-dev pkg-config unzip llvm-5.0-dev clang-5.0 libc++-dev \
      libgflags-dev libgtest-dev libssl-dev libcurl3-dev liblzma-dev \
      libeigen3-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev \
      libgflags-dev libx264-dev libopenjpeg-dev libxvidcore-dev \
      libpng-dev libjpeg-dev libbz2-dev python-pip wget \
      libleveldb-dev libsnappy-dev libhdf5-serial-dev liblmdb-dev python-dev \
      python-tk autoconf autogen libtool libtbb-dev libopenblas-dev \
      liblapacke-dev swig yasm python3.5 python3-pip cpio automake libass-dev \
      libfreetype6-dev libsdl2-dev libtheora-dev libtool \
      libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
      libxcb-xfixes0-dev mercurial texinfo zlib1g-dev curl libcap-dev \
      libgnutls-dev libpq-dev postgresql

RUN apt-get install -y --no-install-recommends libboost-all-dev

# Non-apt-installable dependencies
ENV deps /deps
WORKDIR ${deps}

# CMake (we use 3.7 because >3.8 has issues building OpenCV due to http_proxy)
RUN wget "https://cmake.org/files/v3.12/cmake-3.12.2.tar.gz" && \
    tar -xf cmake-3.12.2.tar.gz && cd ${deps}/cmake-3.12.2 && \
    ./bootstrap --parallel=${cores} -- -DCMAKE_USE_OPENSSL=ON && \
    make -j${cores} && \
    make install && \
    rm -rf ${deps}/cmake-3.12.2.tar.gz ${deps}/cmake-3.12.2

# Python dependencies
WORKDIR /opt/scanner-base
ADD . .
RUN pip3 install -r requirements.txt

ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,video
