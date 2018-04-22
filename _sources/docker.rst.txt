.. _docker:

Docker
======

Docker is a service for managing containers, which you can think of as
lightweight virtual machines. We provide prebuilt
`Docker <https://docs.docker.com>`__ images containing Scanner and all its
dependencies (e.g. OpenCV, Caffe) at
`scannerresearch/scanner <https://hub.docker.com/r/scannerresearch/scanner/>`__.

To start using Scanner with Docker, first install
`Docker <https://docs.docker.com/install/>`__.

If you have a GPU and you're running on Linux, you should also install
`nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__ (which provides GPU
support inside Docker containers).

.. code-block:: bash

   pip install --upgrade nvidia-docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   nvidia-docker-compose pull gpu
   nvidia-docker-compose run --service-ports gpu /bin/bash

Otherwise, you should run:

.. code-block:: bash

   pip install --upgrade docker-compose
   wget https://raw.githubusercontent.com/scanner-research/scanner/master/docker/docker-compose.yml
   docker-compose pull cpu
   docker-compose run --service-ports cpu /bin/bash

If these commands were successful, you should now have bash session inside the
docker container. To start using Scanner to process videos, check out
:ref:`getting-started`.

We also provide several other configurations with the following builds:
* `scannerresearch/scanner:cpu` - CPU-only build
* `scannerresearch/scanner:gpu-8.0-cudnn6` - CUDA 9.1, CUDNN 7
* `scannerresearch/scanner:gpu-8.0-cudnn6` - CUDA 9.0, CUDNN 7
* `scannerresearch/scanner:gpu-8.0-cudnn6` - CUDA 8.0, CUDNN 6
