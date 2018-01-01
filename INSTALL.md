# Building Scanner

*NOTE*: The following build instructions have only been tested on Ubuntu 16.04.

There are five major steps to build and install Scanner:
1 Install apt-get dependencies
2 Install python pip dependencies
3 Run deps.sh to download and install external dependencies
4 Build Scanner
5 Install scannerpy python package

Scanner depends on the following *major* dependencies:

* Python == 2.7
* boost >= 1.63.0
* ffmpeg >= 3.3.1
* opencv >= 3.2.0
* protobuf == 3.4.0
* grpc == 1.7.2
* caffe >= rc5 OR intel-caffe >= 1.0.6

Scanner optionally requires:
* CUDA >= 8.0

Scanner provides a dependency script `deps.sh` to automatically install any or
all of the *major* dependencies if they are not already installed. Each of these
dependencies has a set of required system-level packages. If you need to install
all or most of of these dependencies, run the 'All dependencies' apt-get command
below. If you only need to install a few, we also provide apt-get commands for
each package.

## 1. apt-get Dependencies

All dependencies
```bash
apt-get install \
  build-essential \
  cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev \
  libswscale-dev unzip llvm clang libc++-dev libgflags-dev libgtest-dev \
  libssl-dev libcurl3-dev liblzma-dev libeigen3-dev  \
  libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libgflags-dev \
  libx264-dev libopenjpeg-dev libxvidcore-dev \
  libpng-dev libjpeg-dev libbz2-dev git python-pip wget \
  libleveldb-dev libsnappy-dev libhdf5-serial-dev liblmdb-dev python-dev \
  python-tk autoconf autogen libtool libtbb-dev libopenblas-dev \
  liblapacke-dev swig yasm python2.7 cpio \
  automake libass-dev libfreetype6-dev libsdl2-dev libtheora-dev libtool \
  libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
  libxcb-xfixes0-dev mercurial pkg-config texinfo wget zlib1g-dev \
  curl unzip
```

For gRPC
```bash
apt-get install \
  build-essential autoconf automake libtool curl make g++ \
  unzip clang libc++-dev libgflags-dev libgtest-dev unzip
```

For OpenCV
```bash
apt-get install \
  build-essential cmake git libgtk2.0-dev pkg-config \
  python-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
  libjasper-dev libdc1394-22-dev
```

For FFMPEG
```bash
apt-get -y install \
  build-essential autoconf automake build-essential cmake git libass-dev \
  libfreetype6-dev libsdl2-dev libtheora-dev libtool libva-dev libvdpau-dev \
  libvorbis-dev libxcb1-dev libxcb-shm0-dev libxcb-xfixes0-dev mercurial \
  pkg-config texinfo wget zlib1g-dev
```

## 2. Python Pip Dependencies

Scanner depends on several python packages installable via pip. From the
top-level directory, run:

```bash
pip install -r requirements.txt
```

## 3. Run deps.sh

To install or specify where your *major* dependencies are, from the top-level
directory run:

```bash
bash ./deps.sh
```

This script will query you for each major
dependency and install those that are not already installed. By default,
it will install the dependencies to a local directory inside the scanner repo
(it will not install system-wide).

## 4. Build Scanner

Run the following commands from the top-level directory:
```bash
mkdir build
cd build
cmake ..
make -j
```

## 5. Install scannerpy python package

Run the following commands from the top-level directory:
```bash
python python/setup.py bdist_wheel
pip install dist/scannerpy-0.1.13-py2-none-any.whl
```

Congratulations! You've installed the scannerpy package.
