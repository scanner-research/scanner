.. _from_source:

Building Scanner from source
----------------------------

Scanner provides a dependency script :code:`deps.sh` to automatically install any or all
of its major dependencies if they are not already installed. Each of these
dependencies has a set of required system-level packages.

Scanner depends on the following major dependencies:
  - Python >= 3.5
  - pybind >= 1.58.0
  - ffmpeg >= 3.3.1
  - opencv >= 3.4.0
  - protobuf == 3.4.0
  - grpc == 1.7.2
  - caffe >= rc5 OR intel-caffe >= 1.0.6

To compile with CUDA support, it requires:

  - `CUDA <https://developer.nvidia.com/cuda-downloads>`__ 8.0 or above
  - `cuDNN <https://developer.nvidia.com/cudnn>`__ v6.x or above

Install system-level packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ubuntu 16.04
````````````

Run the following command:

.. code-block:: bash

   apt-get install \
      build-essential \
      cmake git libgtk2.0-dev pkg-config unzip llvm-5.0-dev clang-5.0 libc++-dev \
      libgflags-dev libgtest-dev libssl-dev libcurl3-dev liblzma-dev \
      libeigen3-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev \
      libgflags-dev libx264-dev libopenjpeg-dev libxvidcore-dev \
      libpng-dev libjpeg-dev libbz2-dev wget \
      libleveldb-dev libsnappy-dev libhdf5-serial-dev liblmdb-dev python-dev \
      python-tk autoconf autogen libtool libtbb-dev libopenblas-dev \
      liblapacke-dev swig yasm python3.5 python3-pip cpio automake libass-dev \
      libfreetype6-dev libsdl2-dev libtheora-dev libtool \
      libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
      libxcb-xfixes0-dev mercurial texinfo zlib1g-dev curl libcap-dev \
      libboost-all-dev libgnutls-dev libpq-dev postgresql

macOS
`````

Install `homebrew <https://brew.sh/>`__ then run the following command:

.. code-block:: bash

   brew install coreutils cmake git wget unzip \
                automake fdk-aac lame libass libtool libvorbis libvpx \
                opus sdl shtool texi2html theora x264 x265 xvid nasm \
                eigen glog \
                snappy leveldb gflags glog szip lmdb hdf5 boost boost-python3 \
                llvm python gnutls postgresql libpq libpqxx


Run deps.sh
~~~~~~~~~~~

To install or specify where your major dependencies are, from the top-level directory run:

.. code-block:: bash

   bash ./deps.sh

This script will query you for each major dependency and install those that are not already installed. By default, it will install the dependencies to a local directory inside the scanner repo (it will not install system-wide).

.. note::

   Make sure to follow the directions after :code:`deps.sh` finishes that tell you to
   add entries to your PATH, LD_LIBRARY_PATH, and PYTHONPATH

Build Scanner
~~~~~~~~~~~~~

Run the following commands from the top-level directory:

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make -j$(nproc)

Install scannerpy python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run the following command from the top-level directory:

.. code-block:: bash

   bash ./build.sh

Congratulations! You've installed the scannerpy package. To learn how to start
using Scanner, check out :ref:`getting-started`.
